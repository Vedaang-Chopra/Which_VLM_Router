"""
Configuration system for Artemis Router.

Loads YAML configuration files and provides typed configuration objects.
"""

from dataclasses import dataclass
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class RouterConfig:
    """Router model configuration."""
    checkpoint_path: str
    device: str
    model_name_order: List[str]
    dtype: str
    num_threads: int
    warmup: bool


@dataclass
class DataConfig:
    """Database and data loading configuration."""
    db_url: str
    samples_table: str
    logs_table: str
    id_column: str
    text_column: str
    image_path_column: str
    label_column: str
    split_column: str
    image_root_dir: str
    ares_config_path: Optional[str] = None


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    max_text_length: int
    image_size: int
    use_metadata: bool
    allow_text_only: bool
    allow_image_only: bool


@dataclass
class LoggingConfig:
    """Logging configuration for SQL and W&B."""
    sql_enabled: bool
    wandb_enabled: bool
    wandb_project: str
    wandb_run_name: str
    wandb_entity: Optional[str]
    log_router_probs: bool


@dataclass
class LBConfig:
    """Load balancer interface configuration."""
    enabled: bool
    protocol: str  # "http", "kafka", etc.
    endpoint: str


@dataclass
class TrafficConfig:
    """Traffic simulation configuration."""
    default_rps: float
    default_duration_seconds: int
    synthetic_image_shape: List[int]
    synthetic_text_length: int


@dataclass
class AppConfig:
    """Complete application configuration."""
    router: RouterConfig
    data: DataConfig
    features: FeatureConfig
    logging: LoggingConfig
    lb: LBConfig
    traffic: TrafficConfig


def load_config(path: str) -> AppConfig:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        AppConfig object with all settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is malformed
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    try:
        # Parse router config
        router_cfg = RouterConfig(**raw_config['router'])

        # Parse data config
        data_cfg = DataConfig(**raw_config['data'])

        # Parse feature config
        feature_cfg = FeatureConfig(**raw_config['features'])

        # Parse logging config
        logging_cfg = LoggingConfig(**raw_config['logging'])

        # Parse load balancer config
        lb_cfg = LBConfig(**raw_config['lb'])

        # Parse traffic config
        traffic_cfg = TrafficConfig(**raw_config['traffic'])

        return AppConfig(
            router=router_cfg,
            data=data_cfg,
            features=feature_cfg,
            logging=logging_cfg,
            lb=lb_cfg,
            traffic=traffic_cfg,
        )

    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")
    except TypeError as e:
        raise ValueError(f"Configuration type error: {e}")


def validate_config(config: AppConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.

    Args:
        config: AppConfig to validate

    Returns:
        List of warning/error messages (empty if valid)
    """
    issues = []

    # Validate router checkpoint exists
    checkpoint_path = Path(config.router.checkpoint_path)
    if not checkpoint_path.exists():
        issues.append(f"Router checkpoint not found: {config.router.checkpoint_path}")

    # Validate device
    if config.router.device not in ["cpu", "cuda:0", "cuda", "mps"]:
        if not config.router.device.startswith("cuda:"):
            issues.append(f"Invalid device: {config.router.device}")

    # Validate dtype
    if config.router.dtype not in ["float32", "float16"]:
        issues.append(f"Invalid dtype: {config.router.dtype}")

    # Validate image root directory
    image_root = Path(config.data.image_root_dir)
    if not image_root.exists():
        issues.append(f"Image root directory not found: {config.data.image_root_dir}")

    # Validate model name order
    if len(config.router.model_name_order) == 0:
        issues.append("model_name_order cannot be empty")

    # Validate traffic config
    if config.traffic.default_rps <= 0:
        issues.append("default_rps must be positive")

    if len(config.traffic.synthetic_image_shape) != 3:
        issues.append("synthetic_image_shape must have 3 dimensions [H, W, C]")

    return issues
