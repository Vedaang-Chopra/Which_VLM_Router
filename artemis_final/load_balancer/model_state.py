"""
Model state management for queue and replica tracking.

This module implements the core queueing logic for the load balancer:
- ReplicaState: Tracks when each replica becomes available
- ModelLoadState: Manages all replicas for a model
- ModelStateManager: Orchestrates state for all models

The module supports:
- Simulating request assignments (for candidate evaluation)
- Committing assignments (updating queue state)
- Optional autoscaling based on predicted latency
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .types import SimulationResult
from .config import ModelCapacityConfig, AutoscaleConfig
from .stats_registry import StatsRegistry

logger = logging.getLogger(__name__)


def now_ms() -> float:
    """Get current time in milliseconds."""
    return time.time() * 1000.0


@dataclass
class ReplicaState:
    """
    State for a single replica of a model.

    Attributes:
        available_at_ms: Timestamp when this replica becomes free
        replica_id: Unique identifier for this replica
    """
    available_at_ms: float
    replica_id: int = 0


@dataclass
class ModelLoadState:
    """
    Load state for a single model.

    Manages all replicas for a model and tracks autoscaling metadata.

    Attributes:
        model_name: Name of the model
        replicas: List of replica states
        sla_ms: SLA target for this model
        max_replicas: Maximum number of replicas allowed
        autoscale_config: Autoscaling configuration
        last_scale_time_ms: Last time we scaled (for cooldown)
        total_requests_served: Total requests served by this model
    """
    model_name: str
    replicas: List[ReplicaState] = field(default_factory=list)
    sla_ms: float = 2000.0
    max_replicas: int = 1
    autoscale_config: Optional[AutoscaleConfig] = None
    last_scale_time_ms: float = 0.0
    total_requests_served: int = 0


class ModelStateManager:
    """
    Manages state for all models in the load balancer.

    This class provides methods to:
    - Simulate request assignments without mutating state
    - Commit assignments to update replica availability
    - Optionally autoscale based on predicted latency
    """

    def __init__(
        self,
        model_configs: Dict[str, ModelCapacityConfig],
        stats_registry: StatsRegistry
    ):
        """
        Initialize the model state manager.

        Args:
            model_configs: Dictionary of model configurations
            stats_registry: Registry for per-task/model statistics
        """
        self.model_configs = model_configs
        self.stats_registry = stats_registry
        self.model_states: Dict[str, ModelLoadState] = {}

        # Initialize states for each model
        for model_name, config in model_configs.items():
            replicas = [
                ReplicaState(available_at_ms=0.0, replica_id=i)
                for i in range(config.min_replicas)
            ]

            self.model_states[model_name] = ModelLoadState(
                model_name=model_name,
                replicas=replicas,
                sla_ms=config.sla_ms,
                max_replicas=config.max_replicas,
                autoscale_config=config.autoscale,
                last_scale_time_ms=0.0,
            )

        logger.info(f"Initialized state for {len(self.model_states)} models")

    def simulate_assignment(
        self,
        model_name: str,
        task_type: str,
        arrival_ts_ms: float
    ) -> SimulationResult:
        """
        Simulate assigning a request to a model without mutating state.

        This method is used by the scheduler to evaluate candidates before
        committing to a decision.

        Args:
            model_name: Name of the model
            task_type: Type of task
            arrival_ts_ms: Arrival timestamp in milliseconds

        Returns:
            SimulationResult with predicted latency, cost, accuracy, etc.
        """
        if model_name not in self.model_states:
            raise ValueError(f"Unknown model: {model_name}")

        state = self.model_states[model_name]

        # Find replica with earliest availability
        earliest_replica = min(state.replicas, key=lambda r: r.available_at_ms)
        replica_index = state.replicas.index(earliest_replica)

        # Calculate start time (max of arrival and when replica becomes free)
        start_time_ms = max(arrival_ts_ms, earliest_replica.available_at_ms)
        queue_delay_ms = start_time_ms - arrival_ts_ms

        # Get service time from stats
        service_time_ms = self.stats_registry.estimate_service_time_ms(
            task_type, model_name
        )

        # Check if we had stats
        missing_stats = not self.stats_registry.has_stats(task_type, model_name)

        # Calculate finish time
        finish_time_ms = start_time_ms + service_time_ms
        total_latency_ms = queue_delay_ms + service_time_ms

        # Get cost and accuracy estimates
        est_cost_usd = self.stats_registry.estimate_cost_usd(task_type, model_name)
        est_accuracy = self.stats_registry.estimate_accuracy(task_type, model_name)

        # Calculate queue time before this request
        model_queue_time_before_ms = max(0.0, earliest_replica.available_at_ms - arrival_ts_ms)

        return SimulationResult(
            queue_delay_ms=queue_delay_ms,
            service_time_ms=service_time_ms,
            total_latency_ms=total_latency_ms,
            est_cost_usd=est_cost_usd,
            est_accuracy=est_accuracy,
            model_queue_time_before_ms=model_queue_time_before_ms,
            num_replicas=len(state.replicas),
            finish_time_ms=finish_time_ms,
            replica_index=replica_index,
            missing_stats=missing_stats,
        )

    def commit_assignment(
        self,
        model_name: str,
        sim_result: SimulationResult,
        arrival_ts_ms: float
    ):
        """
        Commit an assignment by updating the replica's availability.

        Args:
            model_name: Name of the model
            sim_result: Result from simulate_assignment
            arrival_ts_ms: Arrival timestamp
        """
        if model_name not in self.model_states:
            raise ValueError(f"Unknown model: {model_name}")

        state = self.model_states[model_name]

        # Update the replica that was selected
        replica_index = sim_result.replica_index
        state.replicas[replica_index].available_at_ms = sim_result.finish_time_ms

        # Increment request counter
        state.total_requests_served += 1

        # Check if we should autoscale
        if state.autoscale_config and state.autoscale_config.enable:
            self._maybe_autoscale(
                model_name,
                sim_result.total_latency_ms,
                arrival_ts_ms
            )

    def _maybe_autoscale(
        self,
        model_name: str,
        predicted_latency_ms: float,
        current_time_ms: float
    ):
        """
        Check if autoscaling is needed and execute if appropriate.

        Args:
            model_name: Name of the model
            predicted_latency_ms: Predicted latency for the current request
            current_time_ms: Current timestamp
        """
        state = self.model_states[model_name]
        config = state.autoscale_config

        if not config or not config.enable:
            return

        # Check cooldown
        time_since_last_scale = current_time_ms - state.last_scale_time_ms
        if time_since_last_scale < config.cooldown_ms:
            return

        # Check if we should scale up
        latency_threshold = state.sla_ms * config.scale_up_latency_factor
        current_replicas = len(state.replicas)

        if predicted_latency_ms > latency_threshold:
            # Scale up if under max replicas
            if current_replicas < state.max_replicas:
                new_replica = ReplicaState(
                    available_at_ms=current_time_ms,
                    replica_id=current_replicas
                )
                state.replicas.append(new_replica)
                state.last_scale_time_ms = current_time_ms

                logger.info(
                    f"Scaled up {model_name}: {current_replicas} -> {current_replicas + 1} replicas "
                    f"(latency={predicted_latency_ms:.1f}ms > threshold={latency_threshold:.1f}ms)"
                )

        # Check if we should scale down (basic utilization-based)
        elif current_replicas > self.model_configs[model_name].min_replicas:
            # Calculate utilization: fraction of replicas that are busy
            busy_replicas = sum(
                1 for r in state.replicas
                if r.available_at_ms > current_time_ms
            )
            utilization = busy_replicas / current_replicas

            if utilization < config.scale_down_util_threshold:
                # Remove the least busy replica
                idle_replicas = [
                    (i, r) for i, r in enumerate(state.replicas)
                    if r.available_at_ms <= current_time_ms
                ]

                if idle_replicas:
                    # Remove first idle replica
                    idx_to_remove = idle_replicas[0][0]
                    state.replicas.pop(idx_to_remove)
                    state.last_scale_time_ms = current_time_ms

                    logger.info(
                        f"Scaled down {model_name}: {current_replicas} -> {current_replicas - 1} replicas "
                        f"(utilization={utilization:.2%} < threshold={config.scale_down_util_threshold:.2%})"
                    )

    def get_state(self, model_name: str) -> ModelLoadState:
        """
        Get the current state for a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelLoadState for the model
        """
        return self.model_states[model_name]

    def get_num_replicas(self, model_name: str) -> int:
        """
        Get the current number of replicas for a model.

        Args:
            model_name: Name of the model

        Returns:
            Number of replicas
        """
        return len(self.model_states[model_name].replicas)

    def get_utilization(self, model_name: str, current_time_ms: float) -> float:
        """
        Get current utilization for a model.

        Args:
            model_name: Name of the model
            current_time_ms: Current timestamp

        Returns:
            Utilization (0.0 to 1.0)
        """
        state = self.model_states[model_name]
        if not state.replicas:
            return 0.0

        busy_replicas = sum(
            1 for r in state.replicas
            if r.available_at_ms > current_time_ms
        )
        return busy_replicas / len(state.replicas)

    def reset(self):
        """Reset all model states to initial configuration."""
        for model_name, config in self.model_configs.items():
            replicas = [
                ReplicaState(available_at_ms=0.0, replica_id=i)
                for i in range(config.min_replicas)
            ]

            self.model_states[model_name] = ModelLoadState(
                model_name=model_name,
                replicas=replicas,
                sla_ms=config.sla_ms,
                max_replicas=config.max_replicas,
                autoscale_config=config.autoscale,
                last_scale_time_ms=0.0,
            )

        logger.info("Reset all model states")

    def get_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics for all models.

        Returns:
            Dictionary with model summaries
        """
        summary = {}
        current_time = now_ms()

        for model_name, state in self.model_states.items():
            summary[model_name] = {
                "num_replicas": len(state.replicas),
                "utilization": self.get_utilization(model_name, current_time),
                "total_requests_served": state.total_requests_served,
                "sla_ms": state.sla_ms,
                "max_replicas": state.max_replicas,
            }

        return summary
