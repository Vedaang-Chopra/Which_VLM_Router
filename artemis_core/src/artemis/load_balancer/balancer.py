import logging
import time
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import  dataclass, field

from .types import RouterOutput, SchedulingContext, SchedulingDecision, SimulationResult, ModelCapacityConfig

logger = logging.getLogger(__name__)

# --- Simplified Helpers ---

@dataclass
class ReplicaState:
    available_at_ms: float
    replica_id: int = 0

@dataclass
class ModelLoadState:
    model_name: str
    replicas: List[ReplicaState] = field(default_factory=list)
    sla_ms: float = 2000.0
    max_replicas: int = 1
    total_requests_served: int = 0

    def estimate_queue_delay(self, arrival_ts_ms: float) -> float:
        if not self.replicas:
            return 0.0
        earliest = min(self.replicas, key=lambda r: r.available_at_ms)
        start_ms = max(arrival_ts_ms, earliest.available_at_ms)
        return start_ms - arrival_ts_ms

class StatsRegistry:
    def __init__(self, stats_path: Optional[str] = None):
        self.stats = {}
        if stats_path:
            try:
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load stats from {stats_path}: {e}")

    def get(self, task_type: str, model_name: str, metric: str, default: float) -> float:
        return self.stats.get(task_type, {}).get(model_name, {}).get(metric, default)

    def estimate_latency(self, task: str, model: str) -> float:
        return self.get(task, model, "avg_latency_ms", 1000.0)

    def estimate_cost(self, task: str, model: str) -> float:
        return self.get(task, model, "cost_per_request_usd", 0.0)

    def estimate_accuracy(self, task: str, model: str) -> float:
        return self.get(task, model, "avg_accuracy", 0.0)

class LoadBalancer:
    def __init__(
        self,
        model_configs: Dict[str, ModelCapacityConfig],
        stats_path: Optional[str] = None,
        mode: str = "capacity_aware",
        max_accuracy_drop: float = 0.05
    ):
        self.stats = StatsRegistry(stats_path)
        self.mode = mode
        self.max_accuracy_drop = max_accuracy_drop
        self.model_configs = model_configs
        
        # Initialize States
        self.states: Dict[str, ModelLoadState] = {}
        for name, cfg in model_configs.items():
            replicas = [ReplicaState(0.0, i) for i in range(cfg.min_replicas)]
            self.states[name] = ModelLoadState(
                model_name=name,
                replicas=replicas,
                sla_ms=cfg.sla_ms,
                max_replicas=cfg.max_replicas
            )
    
    def schedule(self, router_output: RouterOutput, context: SchedulingContext) -> SchedulingDecision:
        # Determine candidates (simplified: usually all models in output)
        candidates = sorted(router_output.router_probs.items(), key=lambda x: x[1], reverse=True)
        candidate_names = [c[0] for c in candidates]
        
        # Logic selection
        if self.mode == "capacity_aware":
            decision = self._schedule_optimized(router_output, context, candidate_names, strategy="capacity")
        elif self.mode == "cost_minimizing":
            decision = self._schedule_optimized(router_output, context, candidate_names, strategy="cost")
        elif self.mode == "balanced":
             decision = self._schedule_optimized(router_output, context, candidate_names, strategy="balanced")
        else:
             # Default fallback to router preference
             decision = self._schedule_router_pref(router_output, context)

        return decision

    def _simulate(self, model_name: str, task_type: str, arrival_ms: float) -> SimulationResult:
        if model_name not in self.states:
            # Fallback for unknown models (e.g. mocked ones)
            logger.warning(f"Model '{model_name}' not found in load balancer states. Using fallback values.")
            return SimulationResult(0, 100, 100, 0, 0, 1, arrival_ms+100, 0, missing_stats=["model_not_found"])
            
        state = self.states[model_name]
        earliest_replica = min(state.replicas, key=lambda r: r.available_at_ms)
        idx = state.replicas.index(earliest_replica)
        
        start_ms = max(arrival_ms, earliest_replica.available_at_ms)
        queue_ms = start_ms - arrival_ms
        service_ms = self.stats.estimate_latency(task_type, model_name)
        
        return SimulationResult(
            queue_delay_ms=queue_ms,
            service_time_ms=service_ms,
            total_latency_ms=queue_ms + service_ms,
            est_cost_usd=self.stats.estimate_cost(task_type, model_name),
            est_accuracy=self.stats.estimate_accuracy(task_type, model_name),
            num_replicas=len(state.replicas),
            finish_time_ms=start_ms + service_ms,
            replica_index=idx
        )

    def _schedule_optimized(self, output: RouterOutput, ctx: SchedulingContext, candidates: List[str], strategy: str) -> SchedulingDecision:
        valid = []
        pref_acc = self.stats.estimate_accuracy(output.task_type, output.preferred_model)

        # Skip accuracy constraint if we have no baseline (pref_acc == 0.0 means no stats)
        enforce_accuracy_drop = pref_acc > 0.0
        
        for model in candidates:
            sim = self._simulate(model, output.task_type, ctx.arrival_ts_ms)
            
            # Constraints
            sla = self.model_configs.get(model, ModelCapacityConfig()).sla_ms
            if sim.total_latency_ms > sla:
                continue
            
            if enforce_accuracy_drop and strategy in ["capacity", "balanced"] and (pref_acc - sim.est_accuracy) > self.max_accuracy_drop:
                continue
                
            valid.append((model, sim))
            
        if not valid:
            return self._schedule_router_pref(output, ctx)
            
        # Optimization Goal
        if strategy == "cost":
            best = min(valid, key=lambda x: x[1].est_cost_usd)
        elif strategy == "balanced":
             # Simple score: accuracy - penalty(cost) - penalty(latency)
             # Normalized roughly: 1.0 acc ~ $0.001 cost ~ 1000ms latency
             best = max(valid, key=lambda x: x[1].est_accuracy - (x[1].est_cost_usd * 100) - (x[1].total_latency_ms / 5000))
        else: 
            # Capacity/Router mode: Pick highest router prob that is valid
            best = max(valid, key=lambda x: output.router_probs.get(x[0], 0))

        return self._commit(output, ctx, best[0], best[1])

    def _schedule_router_pref(self, output: RouterOutput, ctx: SchedulingContext) -> SchedulingDecision:
        model = output.preferred_model
        sim = self._simulate(model, output.task_type, ctx.arrival_ts_ms)
        return self._commit(output, ctx, model, sim, sla_violated=True)

    def _commit(self, output: RouterOutput, ctx: SchedulingContext, model: str, sim: SimulationResult, sla_violated: bool = False) -> SchedulingDecision:
        # Update state
        if model in self.states:
            self.states[model].replicas[sim.replica_index].available_at_ms = sim.finish_time_ms
            self.states[model].total_requests_served += 1
            
        return SchedulingDecision(
            sample_id=output.sample_id,
            task_type=output.task_type,
            chosen_model=model,
            preferred_model=output.preferred_model,
            router_probs=output.router_probs,
            arrival_ts_ms=ctx.arrival_ts_ms,
            total_latency_ms=sim.total_latency_ms,
            est_cost_usd=sim.est_cost_usd,
            est_accuracy=sim.est_accuracy,
            num_replicas=sim.num_replicas,
            sla_violated=sla_violated,
            queue_delay_ms=sim.queue_delay_ms,
            service_time_ms=sim.service_time_ms
        )
