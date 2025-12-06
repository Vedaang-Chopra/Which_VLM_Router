"""
Core type definitions for the Artemis load balancer.

This module provides stable interfaces between all load balancer components:
- RouterOutput: Output from the Artemis router
- SchedulingContext: Context for a scheduling decision
- SchedulingDecision: Result of load balancer scheduling
- RequestMetrics: Extended metrics for logging
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class RouterOutput:
    """
    Output from the Artemis router for a single request.

    Attributes:
        sample_id: Unique identifier for the sample/request
        task_type: Type of task (e.g., "ocr", "chart_vqa")
        router_probs: Probability distribution over models {model_name: probability}
        preferred_model: Model with highest probability (argmax of router_probs)
    """
    sample_id: str
    task_type: str
    router_probs: Dict[str, float]
    preferred_model: str


@dataclass
class SchedulingContext:
    """
    Context information for a scheduling decision.

    Attributes:
        arrival_ts_ms: Arrival timestamp in milliseconds
        load_profile: Name of the load profile (e.g., "low", "medium", "high")
        metadata: Arbitrary metadata about the request
    """
    arrival_ts_ms: float
    load_profile: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulingDecision:
    """
    Result of the load balancer scheduling a request.

    This captures all information about how a request was scheduled,
    including timing, cost, and accuracy estimates.

    Attributes:
        sample_id: Unique identifier for the sample
        task_type: Type of task
        chosen_model: Model selected by the load balancer
        preferred_model: Model preferred by the router (may differ from chosen)
        router_probs: Router's probability distribution over models

        arrival_ts_ms: When the request arrived
        queue_delay_ms: Time spent waiting in queue
        service_time_ms: Estimated service time for the model
        total_latency_ms: queue_delay_ms + service_time_ms

        est_cost_usd: Estimated cost in USD
        est_accuracy: Estimated accuracy for this task/model combination

        model_queue_time_before_ms: Queue time before this request arrived
        num_replicas: Number of replicas for chosen model at decision time

        sla_violated: Whether this request violated the SLA
        accuracy_drop: Accuracy drop vs preferred model (0 if same model chosen)
        missing_stats: Whether stats were missing for this task/model
    """
    sample_id: str
    task_type: str
    chosen_model: str
    preferred_model: str
    router_probs: Dict[str, float]

    arrival_ts_ms: float
    queue_delay_ms: float
    service_time_ms: float
    total_latency_ms: float

    est_cost_usd: float
    est_accuracy: float

    model_queue_time_before_ms: float
    num_replicas: int

    sla_violated: bool = False
    accuracy_drop: float = 0.0
    missing_stats: bool = False


@dataclass
class RequestMetrics:
    """
    Extended metrics for logging, wrapping a SchedulingDecision with experiment context.

    Attributes:
        experiment_name: Name of the experiment
        load_profile: Load profile name
        decision: The scheduling decision
        timestamp_iso: ISO format timestamp (for human readability)
        global_step: Global step counter across the experiment
    """
    experiment_name: str
    load_profile: str
    decision: SchedulingDecision
    timestamp_iso: Optional[str] = None
    global_step: Optional[int] = None


@dataclass
class SimulationResult:
    """
    Result of simulating a request assignment to a model.

    This is used internally by the scheduler to evaluate candidates
    before committing to a decision.

    Attributes:
        queue_delay_ms: Estimated queue delay
        service_time_ms: Estimated service time
        total_latency_ms: queue_delay_ms + service_time_ms
        est_cost_usd: Estimated cost
        est_accuracy: Estimated accuracy
        model_queue_time_before_ms: Queue time before assignment
        num_replicas: Number of replicas
        finish_time_ms: When the request would finish
        replica_index: Index of the replica that would serve this request
        missing_stats: Whether stats were missing
    """
    queue_delay_ms: float
    service_time_ms: float
    total_latency_ms: float
    est_cost_usd: float
    est_accuracy: float
    model_queue_time_before_ms: float
    num_replicas: int
    finish_time_ms: float
    replica_index: int
    missing_stats: bool = False
