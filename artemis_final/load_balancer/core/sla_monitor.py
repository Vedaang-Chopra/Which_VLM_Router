"""
SLA monitoring and metrics computation.

This module provides utilities to compute and track SLA metrics:
- Latency percentiles (p50, p95, p99)
- SLA violation rates
- Cost metrics
- Accuracy metrics

Supports both batch computation (from list of decisions) and
incremental computation (streaming updates).
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

from .types import SchedulingDecision

logger = logging.getLogger(__name__)


@dataclass
class SlaMetrics:
    """
    SLA metrics for a set of requests.

    Attributes:
        latency_p50_ms: 50th percentile latency
        latency_p95_ms: 95th percentile latency
        latency_p99_ms: 99th percentile latency
        violation_rate: Fraction of requests exceeding SLA
        avg_cost_usd: Average cost per request
        total_cost_usd: Total cost across all requests
        avg_accuracy: Average accuracy
        requests: Number of requests
        avg_queue_delay_ms: Average queue delay
        avg_service_time_ms: Average service time
    """
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    violation_rate: float
    avg_cost_usd: float
    total_cost_usd: float
    avg_accuracy: float
    requests: int
    avg_queue_delay_ms: float = 0.0
    avg_service_time_ms: float = 0.0


@dataclass
class DetailedMetrics:
    """
    Detailed metrics broken down by model, task, and load profile.

    Attributes:
        overall: Overall SLA metrics
        by_model: Metrics per model
        by_task: Metrics per task type
        by_load_profile: Metrics per load profile (if available)
        model_usage: Request count per model
        task_distribution: Request count per task type
    """
    overall: SlaMetrics
    by_model: Dict[str, SlaMetrics]
    by_task: Dict[str, SlaMetrics]
    by_load_profile: Dict[str, SlaMetrics]
    model_usage: Dict[str, int]
    task_distribution: Dict[str, int]


def compute_sla_metrics(
    decisions: List[SchedulingDecision],
    latency_sla_ms: float
) -> SlaMetrics:
    """
    Compute SLA metrics from a list of scheduling decisions.

    Args:
        decisions: List of scheduling decisions
        latency_sla_ms: SLA target in milliseconds

    Returns:
        SlaMetrics computed from the decisions
    """
    if not decisions:
        return SlaMetrics(
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            violation_rate=0.0,
            avg_cost_usd=0.0,
            total_cost_usd=0.0,
            avg_accuracy=0.0,
            requests=0,
        )

    # Extract metrics
    latencies = [d.total_latency_ms for d in decisions]
    costs = [d.est_cost_usd for d in decisions]
    accuracies = [d.est_accuracy for d in decisions]
    queue_delays = [d.queue_delay_ms for d in decisions]
    service_times = [d.service_time_ms for d in decisions]

    # Compute percentiles
    latency_p50 = np.percentile(latencies, 50)
    latency_p95 = np.percentile(latencies, 95)
    latency_p99 = np.percentile(latencies, 99)

    # Compute violation rate
    violations = sum(1 for lat in latencies if lat > latency_sla_ms)
    violation_rate = violations / len(decisions)

    # Compute averages
    avg_cost = np.mean(costs)
    total_cost = np.sum(costs)
    avg_accuracy = np.mean(accuracies)
    avg_queue_delay = np.mean(queue_delays)
    avg_service_time = np.mean(service_times)

    return SlaMetrics(
        latency_p50_ms=float(latency_p50),
        latency_p95_ms=float(latency_p95),
        latency_p99_ms=float(latency_p99),
        violation_rate=float(violation_rate),
        avg_cost_usd=float(avg_cost),
        total_cost_usd=float(total_cost),
        avg_accuracy=float(avg_accuracy),
        requests=len(decisions),
        avg_queue_delay_ms=float(avg_queue_delay),
        avg_service_time_ms=float(avg_service_time),
    )


def compute_detailed_metrics(
    decisions: List[SchedulingDecision],
    latency_sla_ms: float,
    load_profile_map: Optional[Dict[str, str]] = None
) -> DetailedMetrics:
    """
    Compute detailed metrics with breakdowns by model, task, and load profile.

    Args:
        decisions: List of scheduling decisions
        latency_sla_ms: SLA target in milliseconds
        load_profile_map: Optional mapping from sample_id to load_profile

    Returns:
        DetailedMetrics with all breakdowns
    """
    # Overall metrics
    overall = compute_sla_metrics(decisions, latency_sla_ms)

    # Group by model
    by_model = defaultdict(list)
    for d in decisions:
        by_model[d.chosen_model].append(d)

    model_metrics = {
        model: compute_sla_metrics(decs, latency_sla_ms)
        for model, decs in by_model.items()
    }

    # Group by task
    by_task = defaultdict(list)
    for d in decisions:
        by_task[d.task_type].append(d)

    task_metrics = {
        task: compute_sla_metrics(decs, latency_sla_ms)
        for task, decs in by_task.items()
    }

    # Group by load profile (if provided)
    load_profile_metrics = {}
    if load_profile_map:
        by_load_profile = defaultdict(list)
        for d in decisions:
            profile = load_profile_map.get(d.sample_id)
            if profile:
                by_load_profile[profile].append(d)

        load_profile_metrics = {
            profile: compute_sla_metrics(decs, latency_sla_ms)
            for profile, decs in by_load_profile.items()
        }

    # Model usage counts
    model_usage = {model: len(decs) for model, decs in by_model.items()}

    # Task distribution
    task_distribution = {task: len(decs) for task, decs in by_task.items()}

    return DetailedMetrics(
        overall=overall,
        by_model=model_metrics,
        by_task=task_metrics,
        by_load_profile=load_profile_metrics,
        model_usage=model_usage,
        task_distribution=task_distribution,
    )


class SlaMonitor:
    """
    Incremental SLA monitor for streaming updates.

    This class maintains running statistics and can provide snapshots
    of current metrics without recomputing from scratch.
    """

    def __init__(self, latency_sla_ms: float):
        """
        Initialize the SLA monitor.

        Args:
            latency_sla_ms: SLA target in milliseconds
        """
        self.latency_sla_ms = latency_sla_ms
        self.decisions: List[SchedulingDecision] = []
        self.load_profile_map: Dict[str, str] = {}

    def update(self, decision: SchedulingDecision, load_profile: Optional[str] = None):
        """
        Add a decision to the monitor.

        Args:
            decision: Scheduling decision to add
            load_profile: Optional load profile name for this decision
        """
        self.decisions.append(decision)
        if load_profile:
            self.load_profile_map[decision.sample_id] = load_profile

    def snapshot(self) -> SlaMetrics:
        """
        Get current SLA metrics.

        Returns:
            SlaMetrics computed from all decisions so far
        """
        return compute_sla_metrics(self.decisions, self.latency_sla_ms)

    def detailed_snapshot(self) -> DetailedMetrics:
        """
        Get detailed metrics with all breakdowns.

        Returns:
            DetailedMetrics with model, task, and load profile breakdowns
        """
        return compute_detailed_metrics(
            self.decisions,
            self.latency_sla_ms,
            self.load_profile_map if self.load_profile_map else None
        )

    def reset(self):
        """Reset the monitor."""
        self.decisions.clear()
        self.load_profile_map.clear()

    def get_violation_rate(self) -> float:
        """Get current violation rate."""
        if not self.decisions:
            return 0.0
        violations = sum(
            1 for d in self.decisions
            if d.total_latency_ms > self.latency_sla_ms
        )
        return violations / len(self.decisions)

    def get_avg_latency(self) -> float:
        """Get average latency."""
        if not self.decisions:
            return 0.0
        return np.mean([d.total_latency_ms for d in self.decisions])

    def get_avg_cost(self) -> float:
        """Get average cost per request."""
        if not self.decisions:
            return 0.0
        return np.mean([d.est_cost_usd for d in self.decisions])

    def get_model_usage_share(self) -> Dict[str, float]:
        """
        Get model usage as fraction of total requests.

        Returns:
            Dictionary mapping model_name to fraction of requests
        """
        if not self.decisions:
            return {}

        counts = defaultdict(int)
        for d in self.decisions:
            counts[d.chosen_model] += 1

        total = len(self.decisions)
        return {model: count / total for model, count in counts.items()}


def print_metrics_summary(metrics: SlaMetrics, name: str = "Overall"):
    """
    Print a formatted summary of SLA metrics.

    Args:
        metrics: SLA metrics to print
        name: Name of the metric group (e.g., "Overall", "small_vlm")
    """
    print(f"\n{name} Metrics:")
    print(f"  Requests:          {metrics.requests}")
    print(f"  Latency p50:       {metrics.latency_p50_ms:.1f} ms")
    print(f"  Latency p95:       {metrics.latency_p95_ms:.1f} ms")
    print(f"  Latency p99:       {metrics.latency_p99_ms:.1f} ms")
    print(f"  Violation rate:    {metrics.violation_rate:.2%}")
    print(f"  Avg cost:          ${metrics.avg_cost_usd:.6f}")
    print(f"  Total cost:        ${metrics.total_cost_usd:.4f}")
    print(f"  Avg accuracy:      {metrics.avg_accuracy:.4f}")
    print(f"  Avg queue delay:   {metrics.avg_queue_delay_ms:.1f} ms")
    print(f"  Avg service time:  {metrics.avg_service_time_ms:.1f} ms")


def print_detailed_summary(detailed: DetailedMetrics):
    """
    Print a comprehensive summary of detailed metrics.

    Args:
        detailed: Detailed metrics to print
    """
    # Overall
    print_metrics_summary(detailed.overall, "Overall")

    # Model breakdown
    print("\n" + "="*60)
    print("Per-Model Breakdown:")
    print("="*60)
    for model, metrics in detailed.by_model.items():
        print_metrics_summary(metrics, f"Model: {model}")

    # Task breakdown
    print("\n" + "="*60)
    print("Per-Task Breakdown:")
    print("="*60)
    for task, metrics in detailed.by_task.items():
        print_metrics_summary(metrics, f"Task: {task}")

    # Load profile breakdown
    if detailed.by_load_profile:
        print("\n" + "="*60)
        print("Per-Load-Profile Breakdown:")
        print("="*60)
        for profile, metrics in detailed.by_load_profile.items():
            print_metrics_summary(metrics, f"Load Profile: {profile}")

    # Model usage
    print("\n" + "="*60)
    print("Model Usage Distribution:")
    print("="*60)
    total = sum(detailed.model_usage.values())
    for model, count in sorted(
        detailed.model_usage.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {model:20s}: {count:6d} requests ({pct:5.1f}%)")
