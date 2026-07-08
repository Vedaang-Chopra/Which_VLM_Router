from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class RouterOutput:
    sample_id: str
    task_type: str
    router_probs: Dict[str, float]
    preferred_model: str
    max_prob: float = 0.0

@dataclass
class SchedulingContext:
    arrival_ts_ms: float
    load_profile: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SchedulingDecision:
    sample_id: str
    task_type: str
    chosen_model: str
    preferred_model: str
    router_probs: Dict[str, float]
    arrival_ts_ms: float
    total_latency_ms: float
    est_cost_usd: float
    est_accuracy: float
    num_replicas: int
    sla_violated: bool = False
    accuracy_drop: float = 0.0
    queue_delay_ms: float = 0.0
    service_time_ms: float = 0.0
    missing_stats: List[str] = field(default_factory=list)

@dataclass
class SimulationResult:
    queue_delay_ms: float
    service_time_ms: float
    total_latency_ms: float
    est_cost_usd: float
    est_accuracy: float
    num_replicas: int
    finish_time_ms: float
    replica_index: int
    missing_stats: List[str] = field(default_factory=list)

@dataclass
class ModelCapacityConfig:
    min_replicas: int = 1
    max_replicas: int = 5
    sla_ms: float = 2000.0
    autoscale: bool = False
