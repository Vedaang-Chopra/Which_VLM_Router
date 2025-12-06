"""
Router Engine - Core inference service for Artemis Router.

Provides high-level APIs for routing samples through the trained router model.
"""

import time
from typing import List, Optional

import torch
from sqlalchemy import create_engine

from .config import AppConfig
from .schemas import Sample, InferenceResult, RouterDecision, LogRecord
from .feature_extractor import FeatureExtractor
from .router_model import load_router
from .db_io import (
    load_sample_from_db,
    load_samples_from_db,
    log_to_sql,
    batch_log_to_sql,
)
from .logging_wandb import (
    init_wandb,
    log_result_to_wandb,
)
from .lb_interface import send_to_lb
from .traffic_simulator import make_synthetic_sample


class RouterEngine:
    """
    High-level router inference engine.

    Handles:
    - Model loading and warmup
    - Feature extraction
    - Single and batch inference
    - Logging to SQL and W&B
    - Load balancer integration
    - Database convenience APIs
    """

    def __init__(self, app_cfg: AppConfig):
        """
        Initialize router engine.

        Args:
            app_cfg: Application configuration

        Raises:
            ValueError: If configuration is invalid
        """
        self.cfg = app_cfg

        # Create database engine
        self.db_engine = create_engine(app_cfg.data.db_url)

        # Initialize feature extractor
        self.fe = FeatureExtractor(
            app_cfg.features,
            app_cfg.router.device,
            app_cfg.router,
        )

        # Load router model
        num_models = len(app_cfg.router.model_name_order)
        self.router = load_router(app_cfg.router, num_models=num_models)

        print(f"[INFO] Router loaded on {app_cfg.router.device}")
        print(f"[INFO] Number of models: {num_models}")
        print(f"[INFO] Model order: {app_cfg.router.model_name_order}")

        # Warmup
        if app_cfg.router.warmup:
            self._warmup()

        # Initialize W&B
        if app_cfg.logging.wandb_enabled:
            init_wandb(app_cfg.logging)
            print("[INFO] W&B logging enabled")

    def _warmup(self):
        """
        Warmup router with a synthetic sample to allocate GPU kernels.
        """
        print("[INFO] Running warmup inference...")
        synthetic_sample = make_synthetic_sample(0, self.cfg.traffic)

        # Run a few warmup iterations
        for _ in range(3):
            _ = self.route_sample(synthetic_sample)

        print("[INFO] Warmup complete")

    def route_sample(self, sample: Sample) -> InferenceResult:
        """
        Route a single sample through the router.

        This is the core routing method that:
        1. Extracts features from the sample
        2. Runs inference through the router
        3. Computes probabilities and selects top-1 model
        4. Logs to SQL and W&B (if enabled)
        5. Sends to load balancer (if enabled)

        Args:
            sample: Input sample

        Returns:
            InferenceResult with router decision
        """
        # Extract features
        features = self.fe.encode_single(sample)

        # Run inference
        start = time.time()
        with torch.no_grad():
            logits = self.router(**features)  # Shape: [1, num_models]
        end = time.time()

        # Post-process
        logits_vec = logits[0].detach().cpu()
        probs_tensor = torch.softmax(logits_vec, dim=-1)
        probs_list = probs_tensor.tolist()

        model_order = self.cfg.router.model_name_order
        probs = {name: float(p) for name, p in zip(model_order, probs_list)}

        chosen_idx = int(probs_tensor.argmax().item())
        chosen_model = model_order[chosen_idx]
        inference_ms = (end - start) * 1000.0

        # Build decision
        decision = RouterDecision(
            sample_id=sample.sample_id,
            chosen_model=chosen_model,
            probs=probs,
            raw_logits=logits_vec.tolist(),
            model_order=model_order,
            inference_ms=inference_ms,
        )

        # Build result
        result = InferenceResult(
            sample=sample,
            router_decision=decision,
        )

        # Logging (SQL)
        if self.cfg.logging.sql_enabled:
            try:
                record = self._build_log_record(result)
                log_to_sql(self.db_engine, self.cfg.data, record)
            except Exception as e:
                print(f"[WARN] Failed to log to SQL: {e}")

        # Logging (W&B)
        if self.cfg.logging.wandb_enabled:
            try:
                log_result_to_wandb(result, log_probs=self.cfg.logging.log_router_probs)
            except Exception as e:
                print(f"[WARN] Failed to log to W&B: {e}")

        # Load balancer
        if self.cfg.lb.enabled:
            try:
                send_to_lb(self.cfg.lb, decision, sample)
            except Exception as e:
                print(f"[WARN] Failed to send to LB: {e}")

        return result

    def route_batch(self, samples: List[Sample]) -> List[InferenceResult]:
        """
        Route multiple samples in a batch.

        Uses batched feature extraction and a single forward pass for efficiency.

        Args:
            samples: List of samples

        Returns:
            List of InferenceResult objects
        """
        if not samples:
            return []

        # Extract features (batched)
        features = self.fe.encode_batch(samples)

        # Run inference
        start = time.time()
        with torch.no_grad():
            logits = self.router(**features)  # Shape: [B, num_models]
        end = time.time()

        # Post-process each sample
        results = []
        batch_inference_ms = (end - start) * 1000.0
        per_sample_ms = batch_inference_ms / len(samples)

        for i, sample in enumerate(samples):
            logits_vec = logits[i].detach().cpu()
            probs_tensor = torch.softmax(logits_vec, dim=-1)
            probs_list = probs_tensor.tolist()

            model_order = self.cfg.router.model_name_order
            probs = {name: float(p) for name, p in zip(model_order, probs_list)}

            chosen_idx = int(probs_tensor.argmax().item())
            chosen_model = model_order[chosen_idx]

            decision = RouterDecision(
                sample_id=sample.sample_id,
                chosen_model=chosen_model,
                probs=probs,
                raw_logits=logits_vec.tolist(),
                model_order=model_order,
                inference_ms=per_sample_ms,
            )

            result = InferenceResult(
                sample=sample,
                router_decision=decision,
            )

            results.append(result)

        # Batch logging to SQL
        if self.cfg.logging.sql_enabled:
            try:
                records = [self._build_log_record(r) for r in results]
                batch_log_to_sql(self.db_engine, self.cfg.data, records)
            except Exception as e:
                print(f"[WARN] Failed to batch log to SQL: {e}")

        # W&B logging (individual)
        if self.cfg.logging.wandb_enabled:
            for result in results:
                try:
                    log_result_to_wandb(result, log_probs=self.cfg.logging.log_router_probs)
                except Exception as e:
                    print(f"[WARN] Failed to log to W&B: {e}")

        # Load balancer (individual)
        if self.cfg.lb.enabled:
            for result in results:
                try:
                    send_to_lb(self.cfg.lb, result.router_decision, result.sample)
                except Exception as e:
                    print(f"[WARN] Failed to send to LB: {e}")

        return results

    def route_by_id(self, sample_id: str, split: Optional[str] = None) -> InferenceResult:
        """
        Load and route a sample from database by ID.

        Args:
            sample_id: Sample identifier
            split: Optional dataset split filter

        Returns:
            InferenceResult
        """
        sample = load_sample_from_db(self.db_engine, self.cfg.data, sample_id, split)
        return self.route_sample(sample)

    def route_split(self, split: str, limit: Optional[int] = None) -> List[InferenceResult]:
        """
        Load and route all samples from a dataset split.

        Args:
            split: Dataset split (e.g., "test", "val")
            limit: Maximum number of samples to process

        Returns:
            List of InferenceResult objects
        """
        samples = load_samples_from_db(self.db_engine, self.cfg.data, split, limit)
        return self.route_batch(samples)

    def _build_log_record(self, result: InferenceResult) -> LogRecord:
        """
        Build a LogRecord from an InferenceResult.

        Args:
            result: Inference result

        Returns:
            LogRecord for SQL insertion
        """
        sample = result.sample
        decision = result.router_decision

        return LogRecord(
            timestamp=time.time(),
            sample_id=sample.sample_id,
            source=sample.source,
            text=sample.text,
            image_uri=sample.image_uri,
            split=sample.metadata.get("split"),
            label=sample.label,
            router_chosen_model=decision.chosen_model,
            router_probs=decision.probs,
            router_inference_ms=decision.inference_ms,
            extra_metadata=sample.metadata,
        )

    def get_stats(self) -> dict:
        """
        Get engine statistics.

        Returns:
            Dictionary with engine stats
        """
        return {
            "device": self.cfg.router.device,
            "dtype": self.cfg.router.dtype,
            "num_models": len(self.cfg.router.model_name_order),
            "model_order": self.cfg.router.model_name_order,
            "sql_logging": self.cfg.logging.sql_enabled,
            "wandb_logging": self.cfg.logging.wandb_enabled,
            "lb_enabled": self.cfg.lb.enabled,
        }
