"""
Core type definitions for the Artemis load balancer.

This module provides stable interfaces between all load balancer components:
- RouterOutput: Output from the Artemis router
- SchedulingContext: Context for a scheduling decision
- SchedulingDecision: Result of load balancer scheduling
- RequestMetrics: Extended metrics for logging
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


class BudgetExhaustedError(Exception):
    """Raised when the global cost budget is exhausted."""
    pass


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
    max_prob: float = 0.0


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
    Result of a load balancer scheduling decision.

    Attributes:
        sample_id: Unique identifier for the sample
        task_type: Task type
        chosen_model: Model selected by the load balancer
        preferred_model: Model preferred by the router
        router_probs: Original router probabilities
        arrival_ts_ms: Timestamp of request arrival
        queue_delay_ms: Estimated time in queue
        service_time_ms: Estimated processing time
        total_latency_ms: queue_delay + service_time
        est_cost_usd: Estimated cost of the request
        est_accuracy: Estimated accuracy of the chosen model
        model_queue_time_before_ms: Queue time of the model before this assignment
        num_replicas: Number of effective replicas for the chosen model
        sla_violated: Whether the estimated latency exceeds the SLA
        accuracy_drop: Difference between preferred model accuracy and chosen model accuracy
        missing_stats: List of stats that were missing for this decision
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
    missing_stats: List[str] = field(default_factory=list)


@dataclass
class RequestMetrics:
    """
    Extended metrics for logging and analysis.
    """
    sample_id: str
    task_type: str
    model: str
    latency_ms: float
    cost_usd: float
    is_correct: Optional[bool] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class SimulationResult:
    """
    Result of a simulation run for a single model assignment.

    Attributes:
        queue_delay_ms: Estimated time the request will spend in queue
        service_time_ms: Estimated processing time
        total_latency_ms: queue_delay + service_time
        est_cost_usd: Estimated cost of the request
        est_accuracy: Estimated accuracy of the model on this task
        model_queue_time_before_ms: Queue time of the model before this assignment
        num_replicas: Number of effective replicas for the model
        finish_time_ms: Estimated timestamp when processing will finish
        replica_index: Index of the assigned replica
        missing_stats: Whether statistics were missing for this estimate
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
    missing_stats: List[str] = field(default_factory=list)

