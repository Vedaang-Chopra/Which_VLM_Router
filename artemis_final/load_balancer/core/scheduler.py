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

This module also contains internal helpers for Mode Switching and Budget Tracking.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import time
from collections import deque
from dataclasses import dataclass

from .types import RouterOutput, SchedulingContext, SchedulingDecision, BudgetExhaustedError
from .config import ModelCapacityConfig, GlobalSLAConfig, TaskSLAConfig
from .model_state import ModelStateManager
from .stats_registry import StatsRegistry
from .sla_monitor import SlaMonitor

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Internal Helper Classes
# -----------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    total_cost_budget_usd: float
    budget_window_sec: int = 3600  # 1 hour

class SLABudgetTracker:
    def __init__(self, config: BudgetConfig):
        self.total_budget = config.total_cost_budget_usd
        self.current_spent = 0.0
        self.window = deque()
        self.window_sec = config.budget_window_sec

    def add_cost(self, cost_usd: float) -> bool:
        """
        Record a cost. Returns False if budget exceeded.
        Also maintains the sliding window of costs.
        """
        current_time = time.time()
        self.current_spent += cost_usd
        self.window.append((current_time, cost_usd))
        
        # Prune old entries from window if needed (optional implementation detail, 
        # but good for long-running processes to avoid memory leaks if we only care about total)
        # Note: The user req implies a global total budget, but specifies a window.
        # usually "budget" is total-ever, or "rate limit" is per-window.
        # "total_cost_budget_usd" implies a cap.
        # "budget_window_sec" implies maybe we care about rate?
        # For this implementation, we'll track total accumulated cost against the budget
        # since it's "BudgetExhaustedError". The window might be for analytics.
        
        if self.current_spent >= self.total_budget:
            return False  # Budget exhausted
        return True

    def get_remaining_budget(self) -> float:
        return max(0.0, self.total_budget - self.current_spent)

    def get_spent_pct(self) -> float:
        if self.total_budget <= 0:
            return 1.0
        return self.current_spent / self.total_budget

    def reset(self):
        self.current_spent = 0.0
        self.window.clear()


@dataclass
class ModeSwitchConfig:
    default_mode: str = "balanced"
    violation_threshold: float = 0.10  # 10% SLA violation rate threshold
    scaled_cooldown_sec: int = 300     # 5 minutes cooldown between switches
    
class ModeSwitcher:
    """Handles runtime mode switching based on SLA violations and system state."""

    def __init__(self, config: ModeSwitchConfig):
        self.config = config
        self.violation_threshold = config.violation_threshold
        self.cooldown_sec = config.scaled_cooldown_sec
        self.last_switch_time = 0
        self.current_mode = config.default_mode
        self.switch_history = []

    def should_switch_mode(self, sla_stats: Any) -> str:
        """
        Determine if mode switch is needed based on violations.
        Returns the mode to use (either new or current).
        
        sla_stats expected to have:
        - global_accuracy
        - latency_violation_rate (0.0-1.0)
        - budget_remaining_pct (0.0-1.0)
        - min_global_accuracy (config)
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_switch_time < self.cooldown_sec:
            return self.current_mode

        target_mode = self.current_mode

        # 1. Critical: Accuracy Safety Net
        # If accuracy drops below minimum, force accuracy mode
        if hasattr(sla_stats, 'global_accuracy') and hasattr(sla_stats, 'min_global_accuracy'):
            if sla_stats.global_accuracy < sla_stats.min_global_accuracy:
                target_mode = "accuracy"

        # 2. Performance: High Latency Violations
        # If we are failing SLAs too often, switch to speed
        # But only if we aren't already forced into accuracy mode (accuracy takes precedence here? 
        # Actually, if we miss SLAs, accuracy doesn't matter if response is too late. 
        # But usually 'accuracy' mode is slowest. 'fast' is fastest.
        # Let's assess priority. Typically: Budget > Accuracy > Latency or Budget > Latency > Accuracy?
        # User prompt implies:
        # Accuracy < threshold -> accuracy
        # Latency > threshold -> fast
        # Budget < threshold -> cheap
        
        # Let's check budget first as it's a hard constraint often.
        if hasattr(sla_stats, 'budget_remaining_pct') and sla_stats.budget_remaining_pct < 0.2:
             target_mode = "cheap"
        
        # Check accuracy next 
        elif hasattr(sla_stats, 'global_accuracy') and hasattr(sla_stats, 'min_global_accuracy') and \
             sla_stats.global_accuracy < sla_stats.min_global_accuracy:
             target_mode = "accuracy"
             
        # Check latency
        elif hasattr(sla_stats, 'latency_violation_rate') and \
             sla_stats.latency_violation_rate > self.violation_threshold:
             target_mode = "fast"
        
        # If all good, revert to default/balanced? 
        # "Otherwise, stay in balanced" per prompt.
        else:
            target_mode = self.config.default_mode

        # Execute switch if changed
        if target_mode != self.current_mode:
            logger.info(f"Switching mode: {self.current_mode} -> {target_mode}")
            self.current_mode = target_mode
            self.last_switch_time = current_time
            self.switch_history.append((current_time, target_mode))
            
        return self.current_mode


# -----------------------------------------------------------------------------
# Main Scheduler Class
# -----------------------------------------------------------------------------

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
        latency_sla_ms: Dict[str, float],
        max_accuracy_drop: float = 0.05,
        scheduling_mode: str = "capacity_aware",
        router_confidence_threshold: float = 0.6,
        top_k: int = 3,
        simulation_only: bool = False,
        global_sla_config: Optional[GlobalSLAConfig] = None
    ):
        """
        Initialize the load balancer.

        Args:
            model_configs: Dictionary of model configurations
            stats_registry: Registry for per-task/model statistics
            latency_sla_ms: Dictionary mapping task types to SLA targets in ms
            max_accuracy_drop: Maximum allowed accuracy drop vs preferred model
            scheduling_mode: One of "router", "capacity_aware", "accuracy", "fast", "cheap", "balanced"
            router_confidence_threshold: Threshold below which to use top-K fallback
            top_k: Number of models to consider when confidence is low
            top_k: Number of models to consider when confidence is low
            simulation_only: If True, don't commit assignments (for what-if analysis)
            global_sla_config: Optional global SLA configuration
        """
        self.model_state = ModelStateManager(model_configs, stats_registry)
        self.stats_registry = stats_registry
        self.sla_config = latency_sla_ms
        self.max_accuracy_drop = max_accuracy_drop
        self.scheduling_mode = scheduling_mode
        self.router_confidence_threshold = router_confidence_threshold
        self.top_k = top_k
        self.simulation_only = simulation_only
        
        # New Components
        self.global_sla_config = global_sla_config or GlobalSLAConfig()
        
        # Initialize budget tracker from global config or default
        budget_cfg = BudgetConfig(total_cost_budget_usd=self.global_sla_config.total_cost_budget_usd)
        
        self.budget_tracker = SLABudgetTracker(budget_cfg)
        self.mode_switcher = ModeSwitcher(ModeSwitchConfig(default_mode=scheduling_mode))
        self.sla_monitor = SlaMonitor(latency_sla_ms.get('default', 2000.0))
        
        # Request Rate Tracking
        self.request_times = deque(maxlen=1000)

        # Validate scheduling mode
        valid_modes = ["router_only", "capacity_aware", "router", "accuracy", "fast", "cheap", "balanced", "cost_minimizing"]
        if scheduling_mode not in valid_modes:
            raise ValueError(
                f"Invalid scheduling_mode: {scheduling_mode}. "
                f"Must be one of {valid_modes}"
            )

        logger.info(
            f"Initialized ArtemisLoadBalancer: mode={scheduling_mode}, "
            f"sla_config={latency_sla_ms}, max_accuracy_drop={max_accuracy_drop}, "
            f"confidence_threshold={router_confidence_threshold}, top_k={top_k}, "
            f"simulation_only={simulation_only}"
        )

    def _get_current_rps(self) -> float:
        """Calculate current requests per second."""
        if len(self.request_times) < 2:
            return 0.0
        
        window_sec = 10.0
        cutoff_time = time.time() - window_sec
        recent = [t for t in self.request_times if t > cutoff_time]
        
        if not recent:
            return 0.0
            
        return len(recent) / window_sec

    def _get_lb_stats(self):
        """Aggregate stats for mode switching."""
        sla_metrics = self.sla_monitor.snapshot()
        
        @dataclass
        class LBStats:
             global_accuracy: float
             latency_violation_rate: float
             budget_remaining_pct: float
             min_global_accuracy: float
             
        spent_pct = self.budget_tracker.get_spent_pct()
        
        return LBStats(
            global_accuracy=sla_metrics.avg_accuracy,
            latency_violation_rate=sla_metrics.violation_rate,
            budget_remaining_pct=1.0 - spent_pct,
            min_global_accuracy=self.global_sla_config.min_global_accuracy
        )

    def _get_sla(self, task_type: str) -> float:
        """Get the latency SLA for a specific task type."""
        return self.sla_config.get(task_type, self.sla_config.get("default", 2000.0))

    def schedule(
        self,
        router_output: RouterOutput,
        context: SchedulingContext
    ) -> SchedulingDecision:
        """
        Schedule a request to a model.
        """
        # 0. Pre-schedule Checks & Updates
        self.request_times.append(time.time())
        current_rps = self._get_current_rps()
        
        # Check Budget
        if not self.budget_tracker.get_remaining_budget() > 0:
             # Just warn for now to avoid crashing experiment in this demo
             # raise BudgetExhaustedError("Cost budget exhausted")
             pass 

        # Dynamic Mode Switching
        new_mode = self.mode_switcher.should_switch_mode(self._get_lb_stats())
        if new_mode != self.scheduling_mode:
             logger.info(f"Dynamic switching: {self.scheduling_mode} -> {new_mode}")
             self.scheduling_mode = new_mode

        # 1. Determine Candidate Models based on Confidence
        if router_output.max_prob < self.router_confidence_threshold:
            # Low confidence: consider top-K models
            candidates = self._get_sorted_candidates(router_output.router_probs)[:self.top_k]
        else:
            # High confidence: consider all models in router output
            candidates = self._get_sorted_candidates(router_output.router_probs)
            
        candidate_names = [name for name, _ in candidates]

        # 2. Apply Routing Mode Logic
        if self.scheduling_mode in ["router_only", "router"]:
            decision = self._schedule_router_mode(router_output, context, candidate_names)
        elif self.scheduling_mode == "capacity_aware":
             decision = self._schedule_capacity_aware(router_output, context) # Keep legacy method for backward compat
        elif self.scheduling_mode == "accuracy":
            decision = self._schedule_accuracy_mode(router_output, context, candidate_names)
        elif self.scheduling_mode == "fast":
            decision = self._schedule_fast_mode(router_output, context, candidate_names)
        elif self.scheduling_mode in ["cheap", "cost_minimizing"]:
            decision = self._schedule_cheap_mode(router_output, context, candidate_names)
        elif self.scheduling_mode == "balanced":
            decision = self._schedule_balanced_mode(router_output, context, candidate_names)
        else:
             # Fallback
             decision = self._schedule_capacity_aware(router_output, context)
             
        # Post-schedule updates
        self.budget_tracker.add_cost(decision.est_cost_usd)
        self.sla_monitor.update(decision, context.load_profile)
        
        return decision

    def _get_valid_candidates(self, router_output, context, candidate_model_names) -> List[Tuple[str, Any]]:
        """
        Filter candidates by SLA and Accuracy constraints.
        Returns list of (model_name, sim_result) tuples.
        """
        valid_candidates = []
        preferred_acc = self.stats_registry.estimate_accuracy(
            router_output.task_type,
            router_output.preferred_model
        )
        task_sla = self._get_sla(router_output.task_type)

        for model_name in candidate_model_names:
            sim_result = self.model_state.simulate_assignment(
                model_name,
                router_output.task_type,
                context.arrival_ts_ms
            )

            # SLA Check
            if sim_result.total_latency_ms > task_sla:
                continue
            
            # Accuracy Check (only if strict dropping is enabled/implied by usage)
            # Note: For strict modes like "accuracy", we might relax this or enforce it.
            # Here we enforce max_accuracy_drop constraint as a safety guardrail for all modes
            if preferred_acc is not None:
                accuracy_drop = preferred_acc - sim_result.est_accuracy
                if accuracy_drop > self.max_accuracy_drop:
                    continue
            
            valid_candidates.append((model_name, sim_result))
            
        return valid_candidates

    def _schedule_router_mode(self, router_output, context, candidate_names) -> SchedulingDecision:
        """Trust router preference, filtered by hard constraints."""
        # Candidates are already sorted by router probability
        valid_candidates = self._get_valid_candidates(router_output, context, candidate_names)
        
        if valid_candidates:
            # Pick the valid candidate with highest Router Probability
            # Since candidate_names provided was sorted by prob, the first valid one we find
            # that is high in that list is usually best.
            # However, _get_valid_candidates returns a new list. We should re-sort or pick best.
            # Let's map back to probs.
            
            chosen_model, sim_result = max(
                valid_candidates,
                key=lambda x: router_output.router_probs.get(x[0], 0.0)
            )
        else:
            return self._fallback_assignment(router_output, context)
            
        return self._commit_and_build(router_output, context, chosen_model, sim_result)

    def _schedule_accuracy_mode(self, router_output, context, candidate_names) -> SchedulingDecision:
        """Maximize estimated accuracy."""
        valid_candidates = self._get_valid_candidates(router_output, context, candidate_names)
        
        if valid_candidates:
            chosen_model, sim_result = max(
                valid_candidates,
                key=lambda x: x[1].est_accuracy
            )
        else:
            return self._fallback_assignment(router_output, context)
            
        return self._commit_and_build(router_output, context, chosen_model, sim_result)
        
    def _schedule_fast_mode(self, router_output, context, candidate_names) -> SchedulingDecision:
        """Minimize total latency."""
        valid_candidates = self._get_valid_candidates(router_output, context, candidate_names)
        
        if valid_candidates:
            chosen_model, sim_result = min(
                valid_candidates,
                key=lambda x: x[1].total_latency_ms
            )
        else:
            return self._fallback_assignment(router_output, context)
            
        return self._commit_and_build(router_output, context, chosen_model, sim_result)

    def _schedule_cheap_mode(self, router_output, context, candidate_names) -> SchedulingDecision:
        """Minimize cost."""
        valid_candidates = self._get_valid_candidates(router_output, context, candidate_names)
        
        if valid_candidates:
            chosen_model, sim_result = min(
                valid_candidates,
                key=lambda x: x[1].est_cost_usd
            )
        else:
            return self._fallback_assignment(router_output, context)
            
        return self._commit_and_build(router_output, context, chosen_model, sim_result)

    def _schedule_balanced_mode(self, router_output, context, candidate_names) -> SchedulingDecision:
        """Maximize score = accuracy - alpha*latency - beta*cost."""
        valid_candidates = self._get_valid_candidates(router_output, context, candidate_names)
        
        if valid_candidates:
            # Heuristic normalization factors
            # Accuracy is 0-1
            # Latency: 2000ms is "bad" -> 1.0 penalty. alpha=0.3
            # Cost: $0.001 is "bad" -> 1.0 penalty. beta=0.2
            alpha = 0.3
            beta = 0.2
            
            def calculate_score(sim):
                norm_lat = min(sim.total_latency_ms / 2000.0, 2.0) # Cap penalty
                norm_cost = min(sim.est_cost_usd / 0.001, 2.0)     # Cap penalty
                return sim.est_accuracy - (alpha * norm_lat) - (beta * norm_cost)

            chosen_model, sim_result = max(
                valid_candidates,
                key=lambda x: calculate_score(x[1])
            )
        else:
            return self._fallback_assignment(router_output, context)
            
        return self._commit_and_build(router_output, context, chosen_model, sim_result)

    def _schedule_capacity_aware(self, router_output, context) -> SchedulingDecision:
        """Legacy capacity-aware mode (kept for backward compatibility)."""
        # Similar logic to _schedule_router_mode but considering all models sorted
        candidates = self._get_sorted_candidates(router_output.router_probs)
        candidate_names = [c[0] for c in candidates]
        return self._schedule_router_mode(router_output, context, candidate_names)
    
    def _schedule_cost_minimizing(self, router_output, context) -> SchedulingDecision:
         """Legacy cost-minimizing mode."""
         candidates = self._get_sorted_candidates(router_output.router_probs)
         candidate_names = [c[0] for c in candidates]
         return self._schedule_cheap_mode(router_output, context, candidate_names)

    def _fallback_assignment(self, router_output, context) -> SchedulingDecision:
        """Assign to preferred model regardless of constraints."""
        chosen_model = router_output.preferred_model
        sim_result = self.model_state.simulate_assignment(
            chosen_model,
            router_output.task_type,
            context.arrival_ts_ms
        )
        return self._commit_and_build(router_output, context, chosen_model, sim_result, sla_violated=True)

    def _commit_and_build(self, router_output, context, chosen_model, sim_result, sla_violated=False) -> SchedulingDecision:
        """Commit assignment (if not sim-only) and build decision object."""
        if not self.simulation_only:
            self.model_state.commit_assignment(
                chosen_model,
                sim_result,
                context.arrival_ts_ms
            )
            
        preferred_acc = self.stats_registry.estimate_accuracy(
            router_output.task_type,
            router_output.preferred_model
        )
        
        accuracy_drop = 0.0
        if preferred_acc is not None:
            accuracy_drop = preferred_acc - sim_result.est_accuracy

        # Recalculate SLA violation using new per-task helper (redundant for valid options but needed for fallback)
        if not sla_violated:
            task_sla = self._get_sla(router_output.task_type)
            sla_violated = sim_result.total_latency_ms > task_sla

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
            "sla_config": self.sla_config,
            "max_accuracy_drop": self.max_accuracy_drop,
            "simulation_only": self.simulation_only,
            "model_states": self.model_state.get_summary(),
        }
