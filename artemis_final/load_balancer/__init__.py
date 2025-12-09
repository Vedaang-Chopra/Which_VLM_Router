"""
Artemis Load Balancer Module

Post-router load balancing and SLA verification for the Artemis VLM routing system.

This module provides:
- SLA-aware scheduling with queue management
- Multiple scheduling policies (router_only, capacity_aware, cost_minimizing)
- Autoscaling simulation
- Comprehensive metrics and logging (W&B, CSV, JSONL)
- Experiment orchestration and analysis tools

Main components:
- ArtemisLoadBalancer: Core scheduler
- ModelStateManager: Queue and replica management
- SlaMonitor: Metrics computation and tracking
- Loggers: CSV, JSONL, and W&B integration
- Evaluation: Experiment runner and analysis templates

Quick start:
    >>> from load_balancer import ArtemisLoadBalancer
    >>> from load_balancer.config import load_capacity_config, default_experiment_config
    >>> from load_balancer.stats_registry import load_per_task_model_stats, StatsRegistry
    >>>
    >>> # Load configurations
    >>> model_configs = load_capacity_config()
    >>> stats_dict = load_per_task_model_stats()
    >>> stats_registry = StatsRegistry(stats_dict)
    >>>
    >>> # Create load balancer
    >>> lb = ArtemisLoadBalancer(
    ...     model_configs=model_configs,
    ...     stats_registry=stats_registry,
    ...     global_latency_sla_ms=2000.0,
    ...     max_accuracy_drop=0.05,
    ...     scheduling_mode="capacity_aware"
    ... )
    >>>
    >>> # Schedule a request
    >>> from load_balancer.types import RouterOutput, SchedulingContext
    >>> router_output = RouterOutput(...)
    >>> context = SchedulingContext(...)
    >>> decision = lb.schedule(router_output, context)

Running experiments:
    From command line:
        python -m load_balancer.evaluation.run_experiment --mode capacity_aware --sla-ms 2000

    Programmatically:
        >>> from load_balancer.evaluation import run_experiment
        >>> results = run_experiment()

Analysis:
    Open load_balancer/evaluation/analysis_template.ipynb for comprehensive analysis.

See individual module docstrings for detailed documentation.
"""

# Core types
from .core.types import (
    RouterOutput,
    SchedulingContext,
    SchedulingDecision,
    RequestMetrics,
    SimulationResult,
)

# Configuration
from .core.config import (
    ExperimentConfig,
    LoadProfileConfig,
    ModelCapacityConfig,
    AutoscaleConfig,
    default_experiment_config,
    load_capacity_config,
    get_output_dir,
)

# Statistics
from .core.stats_registry import (
    StatsRegistry,
    load_per_task_model_stats,
    estimate_service_time_ms,
    estimate_cost_usd,
    estimate_accuracy,
)

# Core components
from .core.scheduler import ArtemisLoadBalancer
from .core.model_state import ModelStateManager, ModelLoadState, ReplicaState
from .core.sla_monitor import (
    SlaMonitor,
    SlaMetrics,
    DetailedMetrics,
    compute_sla_metrics,
    compute_detailed_metrics,
    print_metrics_summary,
    print_detailed_summary,
)

# Loggers
from .core.metrics_logger import (
    CsvMetricsLogger,
    JsonlMetricsLogger,
    load_decisions_from_csv,
    load_decisions_from_jsonl,
)
from .core.wandb_logger import WandbLogger, create_logger as create_wandb_logger

# Public API Facade
from .public_api import (
    ArtemisLoadBalancerModule,
    init_load_balancer,
    schedule_request,
    get_metrics,
    reset_load_balancer_metrics,
    simulate_traffic,
    TrafficSimulationResult,
)

__version__ = "1.0.0"

__all__ = [
    # Public API
    "ArtemisLoadBalancerModule",
    "init_load_balancer",
    "schedule_request",
    "get_metrics",
    "reset_load_balancer_metrics",
    "simulate_traffic",
    "TrafficSimulationResult",

    # Types
    "RouterOutput",
    "SchedulingContext",
    "SchedulingDecision",
    "RequestMetrics",
    "SimulationResult",

    # Configuration
    "ExperimentConfig",
    "LoadProfileConfig",
    "ModelCapacityConfig",
    "AutoscaleConfig",
    "default_experiment_config",
    "load_capacity_config",
    "get_output_dir",

    # Statistics
    "StatsRegistry",
    "load_per_task_model_stats",
    "estimate_service_time_ms",
    "estimate_cost_usd",
    "estimate_accuracy",

    # Core components
    "ArtemisLoadBalancer",
    "ModelStateManager",
    "ModelLoadState",
    "ReplicaState",

    # Monitoring
    "SlaMonitor",
    "SlaMetrics",
    "DetailedMetrics",
    "compute_sla_metrics",
    "compute_detailed_metrics",
    "print_metrics_summary",
    "print_detailed_summary",

    # Loggers
    "CsvMetricsLogger",
    "JsonlMetricsLogger",
    "load_decisions_from_csv",
    "load_decisions_from_jsonl",
    "WandbLogger",
    "create_wandb_logger",
]
