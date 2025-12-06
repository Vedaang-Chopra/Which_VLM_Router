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
from typing import Dict, Optional
import yaml


# Path configuration
BASE_DIR = Path(__file__).parents[1]  # points to artemis_final/
ARES_DIR = BASE_DIR / "ares"
LOAD_BALANCER_DIR = BASE_DIR / "load_balancer"

# Data paths
STATS_PATH = ARES_DIR / "aggregates" / "per_task_model_stats.json"
CAPACITY_CONFIG_PATH = LOAD_BALANCER_DIR / "capacity_config.yaml"

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
class ModelCapacityConfig:
    """Capacity configuration for a single model."""
    model_name: str
    base_latency_ms: float  # average service time without queuing
    min_replicas: int = 1
    max_replicas: int = 1
    sla_ms: float = 2000.0
    max_qps_per_replica: float = 1.0
    autoscale: Optional[AutoscaleConfig] = None


@dataclass
class ExperimentConfig:
    """Configuration for a load balancing experiment."""
    name: str
    load_profiles: Dict[str, LoadProfileConfig]
    global_latency_sla_ms: float
    max_allowed_accuracy_drop: float  # vs router's preferred model
    scheduling_mode: str = "capacity_aware"  # "router_only", "capacity_aware", "cost_minimizing"
    simulation_only: bool = False  # if True, don't commit assignments (for what-if analysis)
    random_seed: Optional[int] = None

    # Logging configuration
    log_to_wandb: bool = True
    log_to_csv: bool = True
    csv_output_dir: Optional[Path] = None

    # Additional metadata
    metadata: Dict = field(default_factory=dict)


def default_experiment_config() -> ExperimentConfig:
    """
    Returns a default experiment configuration with standard load profiles.

    Load profiles:
    - low: 2 QPS for 60 seconds (light load)
    - medium: 10 QPS for 60 seconds (moderate load)
    - high: 30 QPS for 60 seconds (heavy load)
    - burst: 50 QPS for 20 seconds (spike test)
    """
    return ExperimentConfig(
        name="phase5_dynamic_load",
        global_latency_sla_ms=2000.0,
        max_allowed_accuracy_drop=0.05,
        load_profiles={
            "low": LoadProfileConfig("low", qps=2, duration_sec=60),
            "medium": LoadProfileConfig("medium", qps=10, duration_sec=60),
            "high": LoadProfileConfig("high", qps=30, duration_sec=60),
            "burst": LoadProfileConfig("burst", qps=50, duration_sec=20),
        }
    )


def load_capacity_config(config_path: Optional[Path] = None) -> Dict[str, ModelCapacityConfig]:
    """
    Load model capacity configuration from YAML file.

    Args:
        config_path: Path to capacity_config.yaml. If None, uses default path.

    Returns:
        Dictionary mapping model_name to ModelCapacityConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is malformed
    """
    if config_path is None:
        config_path = CAPACITY_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Capacity config not found at {config_path}")

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'models' not in data:
        raise ValueError("Capacity config must contain 'models' key")

    configs = {}
    for model_name, model_data in data['models'].items():
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
            base_latency_ms=model_data['base_latency_ms'],
            min_replicas=model_data.get('min_replicas', 1),
            max_replicas=model_data.get('max_replicas', 1),
            sla_ms=model_data.get('sla_ms', 2000.0),
            max_qps_per_replica=model_data.get('max_qps_per_replica', 1.0),
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
