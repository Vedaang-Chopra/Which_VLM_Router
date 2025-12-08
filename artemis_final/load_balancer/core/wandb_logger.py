"""
Weights & Biases integration for experiment tracking.

This module provides a thin wrapper around W&B for logging:
- Individual scheduling decisions
- Aggregated SLA metrics
- Model usage statistics
- Experiment configuration

All logging is optional and can be disabled via configuration.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available - W&B logging will be disabled")

from .types import SchedulingDecision
from .sla_monitor import SlaMetrics, DetailedMetrics

logger = logging.getLogger(__name__)


class WandbLogger:
    """
    Weights & Biases logger for load balancer experiments.

    Provides methods to log:
    - Per-request decisions
    - Periodic SLA metrics
    - Model usage statistics
    - Experiment summaries
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        enabled: bool = True
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            run_name: Name for this run
            config: Configuration dictionary to log
            entity: W&B entity (username/team)
            tags: Optional list of tags
            enabled: If False, all logging is no-op (for testing)
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        self.step_counter = 0

        if not self.enabled:
            if not WANDB_AVAILABLE:
                logger.warning("W&B not available - logging disabled")
            else:
                logger.info("W&B logging disabled by config")
            return

        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            entity=entity,
            tags=tags or [],
        )

        logger.info(f"Initialized W&B logger: {project}/{run_name}")

    def log_decision(
        self,
        decision: SchedulingDecision,
        load_profile: str,
        step: Optional[int] = None
    ):
        """
        Log a single scheduling decision.

        Args:
            decision: Scheduling decision to log
            load_profile: Load profile name
            step: Optional step number (auto-incremented if not provided)
        """
        if not self.enabled:
            return

        if step is None:
            step = self.step_counter
            self.step_counter += 1

        # Log decision metrics
        wandb.log({
            "step": step,
            "load_profile": load_profile,
            "sample_id": decision.sample_id,
            "task_type": decision.task_type,
            "chosen_model": decision.chosen_model,
            "preferred_model": decision.preferred_model,
            "queue_delay_ms": decision.queue_delay_ms,
            "service_time_ms": decision.service_time_ms,
            "total_latency_ms": decision.total_latency_ms,
            "est_cost_usd": decision.est_cost_usd,
            "est_accuracy": decision.est_accuracy,
            "num_replicas": decision.num_replicas,
            "sla_violated": decision.sla_violated,
            "accuracy_drop": decision.accuracy_drop,
            "model_switched": decision.chosen_model != decision.preferred_model,
        }, step=step)

    def log_sla_metrics(
        self,
        metrics: SlaMetrics,
        load_profile: str,
        stage: str = "overall"
    ):
        """
        Log SLA metrics.

        Args:
            metrics: SLA metrics to log
            load_profile: Load profile name
            stage: Stage identifier (e.g., "overall", "final", "intermediate")
        """
        if not self.enabled:
            return

        prefix = f"{stage}/{load_profile}"

        wandb.log({
            f"{prefix}/latency_p50_ms": metrics.latency_p50_ms,
            f"{prefix}/latency_p95_ms": metrics.latency_p95_ms,
            f"{prefix}/latency_p99_ms": metrics.latency_p99_ms,
            f"{prefix}/violation_rate": metrics.violation_rate,
            f"{prefix}/avg_cost_usd": metrics.avg_cost_usd,
            f"{prefix}/total_cost_usd": metrics.total_cost_usd,
            f"{prefix}/avg_accuracy": metrics.avg_accuracy,
            f"{prefix}/requests": metrics.requests,
            f"{prefix}/avg_queue_delay_ms": metrics.avg_queue_delay_ms,
            f"{prefix}/avg_service_time_ms": metrics.avg_service_time_ms,
        })

    def log_detailed_metrics(
        self,
        detailed: DetailedMetrics,
        stage: str = "final"
    ):
        """
        Log detailed metrics with all breakdowns.

        Args:
            detailed: Detailed metrics to log
            stage: Stage identifier
        """
        if not self.enabled:
            return

        # Overall metrics
        self.log_sla_metrics(detailed.overall, "overall", stage)

        # Per-model metrics
        for model, metrics in detailed.by_model.items():
            prefix = f"{stage}/model_{model}"
            wandb.log({
                f"{prefix}/latency_p50_ms": metrics.latency_p50_ms,
                f"{prefix}/latency_p95_ms": metrics.latency_p95_ms,
                f"{prefix}/violation_rate": metrics.violation_rate,
                f"{prefix}/avg_cost_usd": metrics.avg_cost_usd,
                f"{prefix}/requests": metrics.requests,
            })

        # Per-task metrics
        for task, metrics in detailed.by_task.items():
            prefix = f"{stage}/task_{task}"
            wandb.log({
                f"{prefix}/latency_p50_ms": metrics.latency_p50_ms,
                f"{prefix}/latency_p95_ms": metrics.latency_p95_ms,
                f"{prefix}/violation_rate": metrics.violation_rate,
                f"{prefix}/avg_accuracy": metrics.avg_accuracy,
                f"{prefix}/requests": metrics.requests,
            })

        # Model usage
        total_requests = sum(detailed.model_usage.values())
        for model, count in detailed.model_usage.items():
            share = count / total_requests if total_requests > 0 else 0
            wandb.log({
                f"{stage}/model_usage/{model}_count": count,
                f"{stage}/model_usage/{model}_share": share,
            })

    def log_model_state(
        self,
        model_name: str,
        num_replicas: int,
        utilization: float,
        requests_served: int
    ):
        """
        Log model state for monitoring autoscaling.

        Args:
            model_name: Name of the model
            num_replicas: Current number of replicas
            utilization: Current utilization (0.0 to 1.0)
            requests_served: Total requests served
        """
        if not self.enabled:
            return

        wandb.log({
            f"model_state/{model_name}/replicas": num_replicas,
            f"model_state/{model_name}/utilization": utilization,
            f"model_state/{model_name}/requests_served": requests_served,
        })

    def log_summary(self, summary: Dict[str, Any]):
        """
        Log summary statistics at the end of the run.

        Args:
            summary: Dictionary of summary statistics
        """
        if not self.enabled:
            return

        for key, value in summary.items():
            wandb.run.summary[key] = value

    def log_artifact(self, artifact_path: Path, artifact_type: str, name: str):
        """
        Log a file artifact to W&B.

        Args:
            artifact_path: Path to the artifact file
            artifact_type: Type of artifact (e.g., "dataset", "model", "result")
            name: Name for the artifact
        """
        if not self.enabled:
            return

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(artifact_path))
        wandb.log_artifact(artifact)

        logger.info(f"Logged artifact: {name} ({artifact_type})")

    def log_table(self, name: str, data: list, columns: list):
        """
        Log a table to W&B.

        Args:
            name: Name of the table
            data: List of rows (each row is a list of values)
            columns: List of column names
        """
        if not self.enabled:
            return

        table = wandb.Table(data=data, columns=columns)
        wandb.log({name: table})

    def finish(self):
        """Finish the W&B run."""
        if not self.enabled:
            return

        if self.run:
            wandb.finish()
            logger.info("Finished W&B run")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class DummyWandbLogger:
    """
    Dummy W&B logger that does nothing.

    Useful for testing or when W&B is not available.
    """

    def __init__(self, *args, **kwargs):
        """Initialize dummy logger."""
        pass

    def log_decision(self, *args, **kwargs):
        """No-op."""
        pass

    def log_sla_metrics(self, *args, **kwargs):
        """No-op."""
        pass

    def log_detailed_metrics(self, *args, **kwargs):
        """No-op."""
        pass

    def log_model_state(self, *args, **kwargs):
        """No-op."""
        pass

    def log_summary(self, *args, **kwargs):
        """No-op."""
        pass

    def log_artifact(self, *args, **kwargs):
        """No-op."""
        pass

    def log_table(self, *args, **kwargs):
        """No-op."""
        pass

    def finish(self):
        """No-op."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


def create_logger(
    project: str,
    run_name: str,
    config: Dict[str, Any],
    entity: Optional[str] = None,
    tags: Optional[list] = None,
    enabled: bool = True
) -> WandbLogger:
    """
    Factory function to create a W&B logger.

    Returns a DummyWandbLogger if W&B is not available or disabled.

    Args:
        project: W&B project name
        run_name: Name for this run
        config: Configuration dictionary
        entity: W&B entity
        tags: Optional tags
        enabled: Whether to enable logging

    Returns:
        WandbLogger or DummyWandbLogger
    """
    if not enabled or not WANDB_AVAILABLE:
        return DummyWandbLogger()

    return WandbLogger(
        project=project,
        run_name=run_name,
        config=config,
        entity=entity,
        tags=tags,
        enabled=enabled
    )
