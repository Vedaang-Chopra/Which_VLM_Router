"""
Artemis load balancer scheduler.

This module implements the core load balancer that takes router predictions
and makes final scheduling decisions based on:
- SLA constraints
- Queue state
- Accuracy requirements
- Cost optimization (in cost_minimizing mode)

The scheduler supports multiple modes:
- router_only: Always use router's preferred model (baseline)
- capacity_aware: SLA and accuracy-aware scheduling (default)
- cost_minimizing: Minimize cost while satisfying constraints
"""

import logging
from typing import Dict, List, Tuple, Optional

from .types import RouterOutput, SchedulingContext, SchedulingDecision
from .config import ModelCapacityConfig
from .model_state import ModelStateManager
from .stats_registry import StatsRegistry

logger = logging.getLogger(__name__)


class ArtemisLoadBalancer:
    """
    Main load balancer for Artemis.

    This class implements the post-router scheduling logic that selects
    the final model to serve each request based on capacity, SLA, and
    accuracy constraints.
    """

    def __init__(
        self,
        model_configs: Dict[str, ModelCapacityConfig],
        stats_registry: StatsRegistry,
        global_latency_sla_ms: float,
        max_accuracy_drop: float = 0.05,
        scheduling_mode: str = "capacity_aware",
        simulation_only: bool = False
    ):
        """
        Initialize the load balancer.

        Args:
            model_configs: Dictionary of model configurations
            stats_registry: Registry for per-task/model statistics
            global_latency_sla_ms: Global SLA target in milliseconds
            max_accuracy_drop: Maximum allowed accuracy drop vs preferred model
            scheduling_mode: One of "router_only", "capacity_aware", "cost_minimizing"
            simulation_only: If True, don't commit assignments (for what-if analysis)
        """
        self.model_state = ModelStateManager(model_configs, stats_registry)
        self.stats_registry = stats_registry
        self.global_latency_sla_ms = global_latency_sla_ms
        self.max_accuracy_drop = max_accuracy_drop
        self.scheduling_mode = scheduling_mode
        self.simulation_only = simulation_only

        # Validate scheduling mode
        valid_modes = ["router_only", "capacity_aware", "cost_minimizing"]
        if scheduling_mode not in valid_modes:
            raise ValueError(
                f"Invalid scheduling_mode: {scheduling_mode}. "
                f"Must be one of {valid_modes}"
            )

        logger.info(
            f"Initialized ArtemisLoadBalancer: mode={scheduling_mode}, "
            f"sla={global_latency_sla_ms}ms, max_accuracy_drop={max_accuracy_drop}, "
            f"simulation_only={simulation_only}"
        )

    def schedule(
        self,
        router_output: RouterOutput,
        context: SchedulingContext
    ) -> SchedulingDecision:
        """
        Schedule a request to a model.

        This is the main entry point for the load balancer. It evaluates
        candidate models and selects the best one based on the configured
        scheduling policy.

        Args:
            router_output: Output from the Artemis router
            context: Scheduling context with arrival time and metadata

        Returns:
            SchedulingDecision with chosen model and all metrics
        """
        if self.scheduling_mode == "router_only":
            return self._schedule_router_only(router_output, context)
        elif self.scheduling_mode == "capacity_aware":
            return self._schedule_capacity_aware(router_output, context)
        elif self.scheduling_mode == "cost_minimizing":
            return self._schedule_cost_minimizing(router_output, context)
        else:
            raise ValueError(f"Unknown scheduling mode: {self.scheduling_mode}")

    def _schedule_router_only(
        self,
        router_output: RouterOutput,
        context: SchedulingContext
    ) -> SchedulingDecision:
        """
        Baseline mode: always use router's preferred model.

        Args:
            router_output: Router output
            context: Scheduling context

        Returns:
            SchedulingDecision
        """
        chosen_model = router_output.preferred_model

        # Simulate assignment
        sim_result = self.model_state.simulate_assignment(
            chosen_model,
            router_output.task_type,
            context.arrival_ts_ms
        )

        # Commit if not simulation-only
        if not self.simulation_only:
            self.model_state.commit_assignment(
                chosen_model,
                sim_result,
                context.arrival_ts_ms
            )

        # Build decision
        return self._build_decision(
            router_output,
            context,
            chosen_model,
            sim_result
        )

    def _schedule_capacity_aware(
        self,
        router_output: RouterOutput,
        context: SchedulingContext
    ) -> SchedulingDecision:
        """
        Capacity-aware mode: consider SLA and accuracy constraints.

        Policy:
        1. Get accuracy of preferred model
        2. Sort models by router probability (descending)
        3. For each candidate:
           - Simulate assignment
           - Check SLA constraint
           - Check accuracy constraint
           - Pick first that satisfies both
        4. If none satisfy, fall back to preferred model

        Args:
            router_output: Router output
            context: Scheduling context

        Returns:
            SchedulingDecision
        """
        # Get preferred model accuracy
        preferred_acc = self.stats_registry.estimate_accuracy(
            router_output.task_type,
            router_output.preferred_model
        )

        # Sort candidates by router probability
        candidates = self._get_sorted_candidates(router_output.router_probs)

        # Try each candidate
        for model_name, prob in candidates:
            sim_result = self.model_state.simulate_assignment(
                model_name,
                router_output.task_type,
                context.arrival_ts_ms
            )

            # Check SLA constraint
            if sim_result.total_latency_ms > self.global_latency_sla_ms:
                continue

            # Check accuracy constraint
            accuracy_drop = preferred_acc - sim_result.est_accuracy
            if accuracy_drop > self.max_accuracy_drop:
                continue

            # This candidate satisfies constraints
            chosen_model = model_name
            break
        else:
            # No candidate satisfied constraints, fall back to preferred
            chosen_model = router_output.preferred_model
            sim_result = self.model_state.simulate_assignment(
                chosen_model,
                router_output.task_type,
                context.arrival_ts_ms
            )

        # Commit if not simulation-only
        if not self.simulation_only:
            self.model_state.commit_assignment(
                chosen_model,
                sim_result,
                context.arrival_ts_ms
            )

        # Build decision
        return self._build_decision(
            router_output,
            context,
            chosen_model,
            sim_result,
            preferred_acc=preferred_acc
        )

    def _schedule_cost_minimizing(
        self,
        router_output: RouterOutput,
        context: SchedulingContext
    ) -> SchedulingDecision:
        """
        Cost-minimizing mode: choose cheapest model that satisfies constraints.

        Policy:
        1. Get accuracy of preferred model
        2. Evaluate all models
        3. Filter by SLA and accuracy constraints
        4. Among valid candidates, pick cheapest
        5. If none valid, fall back to preferred model

        Args:
            router_output: Router output
            context: Scheduling context

        Returns:
            SchedulingDecision
        """
        # Get preferred model accuracy
        preferred_acc = self.stats_registry.estimate_accuracy(
            router_output.task_type,
            router_output.preferred_model
        )

        # Evaluate all models in router output
        valid_candidates = []

        for model_name, prob in router_output.router_probs.items():
            sim_result = self.model_state.simulate_assignment(
                model_name,
                router_output.task_type,
                context.arrival_ts_ms
            )

            # Check SLA constraint
            if sim_result.total_latency_ms > self.global_latency_sla_ms:
                continue

            # Check accuracy constraint
            accuracy_drop = preferred_acc - sim_result.est_accuracy
            if accuracy_drop > self.max_accuracy_drop:
                continue

            # Valid candidate
            valid_candidates.append((model_name, sim_result))

        # Pick cheapest among valid candidates
        if valid_candidates:
            chosen_model, sim_result = min(
                valid_candidates,
                key=lambda x: x[1].est_cost_usd
            )
        else:
            # Fall back to preferred model
            chosen_model = router_output.preferred_model
            sim_result = self.model_state.simulate_assignment(
                chosen_model,
                router_output.task_type,
                context.arrival_ts_ms
            )

        # Commit if not simulation-only
        if not self.simulation_only:
            self.model_state.commit_assignment(
                chosen_model,
                sim_result,
                context.arrival_ts_ms
            )

        # Build decision
        return self._build_decision(
            router_output,
            context,
            chosen_model,
            sim_result,
            preferred_acc=preferred_acc
        )

    def _build_decision(
        self,
        router_output: RouterOutput,
        context: SchedulingContext,
        chosen_model: str,
        sim_result,
        preferred_acc: Optional[float] = None
    ) -> SchedulingDecision:
        """
        Build a SchedulingDecision from simulation result.

        Args:
            router_output: Router output
            context: Scheduling context
            chosen_model: Model that was chosen
            sim_result: SimulationResult from model_state
            preferred_acc: Accuracy of preferred model (if available)

        Returns:
            SchedulingDecision
        """
        # Calculate accuracy drop
        if preferred_acc is not None:
            accuracy_drop = preferred_acc - sim_result.est_accuracy
        else:
            accuracy_drop = 0.0

        # Check SLA violation
        sla_violated = sim_result.total_latency_ms > self.global_latency_sla_ms

        return SchedulingDecision(
            sample_id=router_output.sample_id,
            task_type=router_output.task_type,
            chosen_model=chosen_model,
            preferred_model=router_output.preferred_model,
            router_probs=router_output.router_probs,
            arrival_ts_ms=context.arrival_ts_ms,
            queue_delay_ms=sim_result.queue_delay_ms,
            service_time_ms=sim_result.service_time_ms,
            total_latency_ms=sim_result.total_latency_ms,
            est_cost_usd=sim_result.est_cost_usd,
            est_accuracy=sim_result.est_accuracy,
            model_queue_time_before_ms=sim_result.model_queue_time_before_ms,
            num_replicas=sim_result.num_replicas,
            sla_violated=sla_violated,
            accuracy_drop=accuracy_drop,
            missing_stats=sim_result.missing_stats,
        )

    def _get_sorted_candidates(
        self,
        router_probs: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Get models sorted by router probability (descending).

        Args:
            router_probs: Dictionary of {model_name: probability}

        Returns:
            List of (model_name, probability) tuples sorted descending
        """
        return sorted(
            router_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def reset(self):
        """Reset the load balancer state."""
        self.model_state.reset()
        logger.info("Reset load balancer state")

    def get_summary(self) -> Dict:
        """
        Get summary statistics for the load balancer.

        Returns:
            Dictionary with summary statistics
        """
        return {
            "scheduling_mode": self.scheduling_mode,
            "global_latency_sla_ms": self.global_latency_sla_ms,
            "max_accuracy_drop": self.max_accuracy_drop,
            "simulation_only": self.simulation_only,
            "model_states": self.model_state.get_summary(),
        }
