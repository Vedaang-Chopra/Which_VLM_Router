"""
Weights & Biases logging for Artemis Router.

Handles W&B initialization and per-request logging.
"""

import wandb
from typing import Optional
from .config import LoggingConfig
from .schemas import InferenceResult


_wandb_initialized = False


def init_wandb(cfg: LoggingConfig):
    """
    Initialize a W&B run for router logging.

    Args:
        cfg: Logging configuration
    """
    global _wandb_initialized

    if _wandb_initialized:
        print("[INFO] W&B already initialized, skipping re-initialization")
        return

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        entity=cfg.wandb_entity,
        config={
            "phase": "router_live_inference",
            "log_router_probs": cfg.log_router_probs,
        },
    )

    _wandb_initialized = True


def log_result_to_wandb(result: InferenceResult, log_probs: bool = True):
    """
    Log a single inference result to W&B.

    Args:
        result: Inference result to log
        log_probs: Whether to log full probability distribution
    """
    if not _wandb_initialized:
        print("[WARN] W&B not initialized, skipping logging")
        return

    s = result.sample
    d = result.router_decision

    payload = {
        "sample_id": s.sample_id,
        "source": s.source,
        "split": s.metadata.get("split"),
        "router/chosen_model": d.chosen_model,
        "router/latency_ms": d.inference_ms,
    }

    # Add label information if available
    if s.label is not None:
        payload["router/has_label"] = True
        payload["label"] = s.label

    # Add router task if available
    if "router_task" in s.metadata:
        payload["router_task"] = s.metadata["router_task"]

    # Add probability distribution if requested
    if log_probs:
        for model_name, prob in d.probs.items():
            payload[f"router/prob_{model_name}"] = prob

    # Add chosen model probability
    payload["router/chosen_prob"] = d.probs[d.chosen_model]

    wandb.log(payload)


def finish_wandb():
    """Finish the W&B run."""
    global _wandb_initialized

    if _wandb_initialized:
        wandb.finish()
        _wandb_initialized = False
