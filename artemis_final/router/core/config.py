"""
Router Artemis Configuration Classes

Defines configuration dataclasses used across router modules:
- TrafficConfig: Traffic simulation parameters
- LoggingConfig: SQL and W&B logging settings
- LBConfig: Load balancer interface configuration

These are used by:
- traffic_simulator.py
- logging_wandb.py
- lb_interface.py
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TrafficConfig:
    """Configuration for traffic simulation and load testing"""

    # Requests per second
    default_rps: float = 10.0

    # Simulation duration in seconds
    default_duration_seconds: int = 60

    # Synthetic image generation
    synthetic_image_shape: tuple = (224, 224, 3)

    # Synthetic text generation
    synthetic_text_length: int = 32  # words

    # Request distribution
    arrival_pattern: str = "poisson"  # poisson | uniform | bursty

    # Sample selection
    sample_selection_strategy: str = "random"  # random | sequential | stratified


@dataclass
class LoggingConfig:
    """Configuration for logging router decisions"""

    # SQL Database Logging
    sql_enabled: bool = True
    db_url: Optional[str] = None  # Defaults to router config db_url
    logs_table: str = "router_live_logs"

    # Weights & Biases Logging
    wandb_enabled: bool = False
    wandb_project: str = "artemis-router-inference"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: list = field(default_factory=list)

    # What to log
    log_router_probs: bool = True  # Log full reward distribution
    log_metadata: bool = True  # Log task, dataset, etc.
    log_latency: bool = True  # Log router inference time

    # Batch logging (for performance)
    batch_size: int = 100  # Flush to DB every N requests
    flush_interval_seconds: int = 30  # Or flush every N seconds


@dataclass
class LBConfig:
    """Configuration for load balancer interface"""

    # Enable/disable load balancer integration
    enabled: bool = False

    # Communication protocol
    protocol: str = "http"  # http | grpc | kafka

    # Load balancer endpoint
    endpoint: str = "http://localhost:8000/route"

    # Timeout settings
    timeout_seconds: float = 5.0
    retry_attempts: int = 3

    # Fallback behavior on LB failure
    fallback_to_router_choice: bool = True

    # Headers / authentication
    headers: Dict[str, str] = field(default_factory=dict)
    api_key: Optional[str] = None

    # Health check
    health_check_interval_seconds: int = 60
    health_check_endpoint: str = "/health"
