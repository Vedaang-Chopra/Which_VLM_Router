"""
Central configuration for the Artemis load balancer module.

This module provides configuration structures and functions for:
- Global SLA settings
- Per-task model statistics paths
- W&B project configuration
- Experiment configurations (load profiles, seeds, etc.)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any
import yaml
import sys

# Import unified config loader
# Need to ensure artemis_final is in path
try:
    from artemis_final.common.config_loader import load_global_config, get_models_config
except ImportError:
    # Fallback if running relative
    import sys
    sys.path.append(str(Path(__file__).parents[3]))
    from artemis_final.common.config_loader import load_global_config, get_models_config

# Path configuration
BASE_DIR = Path(__file__).parents[2]  # points to artemis_final/
ARES_DIR = BASE_DIR / "ares"
LOAD_BALANCER_DIR = BASE_DIR / "load_balancer"

# Data paths
STATS_PATH = ARES_DIR / "aggregates" / "per_task_model_stats.json"
# Deprecated: CAPACITY_CONFIG_PATH = LOAD_BALANCER_DIR / "load_balancer_config.yaml"
# Now we point to the global config just for reference if needed, but we use the loader
CAPACITY_CONFIG_PATH = BASE_DIR / "configs" / "artemis.yaml"

# W&B configuration
WANDB_PROJECT = "artemis_load_balancer"
WANDB_ENTITY = None  # optional, set if needed


@dataclass
class LoadProfileConfig:
    """Configuration for a single load profile."""
    name: str
    qps: float  # queries per second
    duration_sec: int  # duration of this load profile


@dataclass
class AutoscaleConfig:
    """Autoscaling configuration for a model."""
    enable: bool = True
    scale_up_latency_factor: float = 0.8  # scale up if predicted > this * sla_ms
    scale_down_util_threshold: float = 0.3  # scale down if utilization < this
    cooldown_ms: float = 60000  # minimum time between scaling operations


@dataclass
class TaskSLAConfig:
    """Per-task SLA configuration"""
    task_type: str
    max_latency_ms: int
    min_accuracy: float = 0.85


@dataclass
class GlobalSLAConfig:
    """Global SLA settings"""
    total_cost_budget_usd: float = 10.0
    min_global_accuracy: float = 0.85
    default_latency_ms: int = 2000
    task_slas: Dict[str, TaskSLAConfig] = field(default_factory=dict)


@dataclass
class ModelCapacityConfig:
    """Capacity configuration for a single model."""
    model_name: str
    base_latency_ms: float  # average service time without queuing
    min_replicas: int = 1
    max_replicas: int = 1
    sla_ms: float = 2000.0
    max_qps_per_replica: float = 1.0
    cost_per_request_usd: float = 0.0001  # cost estimate for cost_minimizing mode
    autoscale: Optional[AutoscaleConfig] = None


@dataclass
class ExperimentConfig:
    """Configuration for a load balancing experiment."""
    name: str
    load_profiles: Dict[str, LoadProfileConfig]
    
    
    # SLA & Constraints
    global_sla: GlobalSLAConfig = field(default_factory=GlobalSLAConfig)
    latency_sla_ms: Dict[str, float] = field(default_factory=dict) # Legacy shim
    max_allowed_accuracy_drop: float = 0.05
    cost_budget_usd: Optional[float] = None
    min_global_accuracy: float = 0.85

    # Routing Configuration
    scheduling_mode: str = "capacity_aware"  # router, capacity_aware, cost_minimizing, accuracy, fast, cheap, balanced
    router_confidence_threshold: float = 0.6
    top_k: int = 3
    
    simulation_only: bool = False  # if True, don't commit assignments
    random_seed: Optional[int] = None

    # Logging configuration
    log_to_wandb: bool = True
    log_to_csv: bool = True
    csv_output_dir: Optional[Path] = None

    # Additional metadata
    metadata: Dict = field(default_factory=dict)


def get_latency_sla(config: ExperimentConfig, task_type: str) -> float:
    """Get latency SLA for a specific task type, falling back to default."""
    return config.latency_sla_ms.get(task_type, config.latency_sla_ms.get("default", 2000.0))


def default_experiment_config() -> ExperimentConfig:
    """
    Returns a default experiment configuration with standard load profiles.
    """
    # Load global settings to populate defaults
    try:
        cfg = load_global_config()
        lb_cfg = cfg.load_balancer
        g_sla = lb_cfg.global_sla
        
        # Convert task SLAs dict to what GlobalSLAConfig expects if needed, 
        # or just use what we have.
        # GlobalSLAConfig.task_slas expect Dict[str, TaskSLAConfig]
        # but config loader returns Dict[str, Dict]
        task_slas_obj = {}
        for t, params in lb_cfg.task_slas.items():
            task_slas_obj[t] = TaskSLAConfig(
                task_type=t,
                max_latency_ms=int(params.get("max_latency_ms", 2000)),
                min_accuracy=float(params.get("min_accuracy", 0.85))
            )
            
        global_sla = GlobalSLAConfig(
            total_cost_budget_usd=g_sla.total_cost_budget_usd,
            min_global_accuracy=g_sla.min_global_accuracy,
            default_latency_ms=g_sla.default_latency_ms,
            task_slas=task_slas_obj
        )
        
        max_drop = lb_cfg.max_accuracy_drop
        sched_mode = lb_cfg.default_scheduling_mode
        
    except Exception as e:
        # Fallback to defaults if config load fails
        print(f"Warning: Could not load global config for defaults: {e}")
        global_sla = GlobalSLAConfig()
        max_drop = 0.05
        sched_mode = "capacity_aware"

    return ExperimentConfig(
        name="phase5_dynamic_load",
        global_sla=global_sla,
        latency_sla_ms={"default": float(global_sla.default_latency_ms)},
        max_allowed_accuracy_drop=max_drop,
        scheduling_mode=sched_mode,
        load_profiles={
            "low": LoadProfileConfig("low", qps=2, duration_sec=60),
            "medium": LoadProfileConfig("medium", qps=10, duration_sec=60),
            "high": LoadProfileConfig("high", qps=30, duration_sec=60),
            "burst": LoadProfileConfig("burst", qps=50, duration_sec=20),
        }
    )


def load_capacity_config(config_path: Optional[Path] = None) -> Dict[str, ModelCapacityConfig]:
    """
    Load model capacity configuration from Unified Artemis Config.

    Args:
        config_path: Ignored/Optional (kept for signature compatibility), logic uses global loader.

    Returns:
        Dictionary mapping model_name to ModelCapacityConfig
    """
    # Use global config loader
    cfg = load_global_config()
    models_list = get_models_config(cfg)
    
    configs = {}
    for model_data in models_list:
        model_name = model_data.get("name")
        if not model_name:
            continue
            
        # Deduplicate: Only load if not already loaded (assuming first entry has relevant stats)
        # Or if the current one has 'base_latency_ms', we prefer that.
        if model_name in configs:
            continue
            
        # Check if this model entry has load balancer stats
        if "base_latency_ms" not in model_data:
            # Skip entries that don't have LB config (e.g. pure API definitions without simulation params)
            continue
            
        # Parse autoscale config if present
        autoscale = None
        if 'autoscale' in model_data:
            autoscale_data = model_data['autoscale']
            autoscale = AutoscaleConfig(
                enable=autoscale_data.get('enable', True),
                scale_up_latency_factor=autoscale_data.get('scale_up_latency_factor', 0.8),
                scale_down_util_threshold=autoscale_data.get('scale_down_util_threshold', 0.3),
                cooldown_ms=autoscale_data.get('cooldown_ms', 60000),
            )

        configs[model_name] = ModelCapacityConfig(
            model_name=model_name,
            base_latency_ms=model_data.get('base_latency_ms', 1000.0),
            min_replicas=model_data.get('min_replicas', 1),
            max_replicas=model_data.get('max_replicas', 1),
            sla_ms=model_data.get('sla_ms', 2000.0),
            max_qps_per_replica=model_data.get('max_qps_per_replica', 1.0),
            cost_per_request_usd=model_data.get('cost_per_request_usd', 0.0001),
            autoscale=autoscale,
        )

    return configs


def get_output_dir(experiment_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    Get output directory for experiment logs and artifacts.

    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for outputs. If None, uses load_balancer/outputs/

    Returns:
        Path to experiment output directory (created if doesn't exist)
    """
    if base_dir is None:
        base_dir = LOAD_BALANCER_DIR / "outputs"

    output_dir = base_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
