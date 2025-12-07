"""
Unified configuration loader for the Artemis VLM Router system.
Loads from configs/artemis.yaml and provides typed access to all config sections.
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class DBConfig:
    url: str

@dataclass
class RouterConfig:
    checkpoint_path: str
    config_file: str
    device: str = "cpu"

@dataclass
class LoadBalancerConfig:
    config_file: str
    # Populated after loading the referenced file
    global_sla_ms: int = 2000
    models: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceConfig:
    models_file: str

@dataclass
class DataCollectionConfig:
    samples_table: str
    responses_table: str
    feedback_table: str

@dataclass
class RetrainingConfig:
    epochs: int = 1
    batch_size: int = 8
    output_checkpoint: str = "checkpoints/best_reward_router_updated.pt"

@dataclass
class GlobalConfig:
    """Master configuration for the entire Artemis system."""
    db: DBConfig
    router: RouterConfig
    load_balancer: LoadBalancerConfig
    inference: InferenceConfig
    data_collection: DataCollectionConfig
    retraining: RetrainingConfig
    _base_dir: str = ""

def get_base_dir() -> Path:
    """Get the base directory of the artemis_final project."""
    # Assumes common/config_loader.py is at artemis_final/common/
    return Path(__file__).parent.parent.resolve()

def load_global_config(config_path: Optional[str] = None) -> GlobalConfig:
    """
    Load the unified configuration from configs/artemis.yaml.
    
    Args:
        config_path: Optional override for config file path.
        
    Returns:
        GlobalConfig dataclass with all settings.
    """
    base_dir = get_base_dir()
    
    if config_path is None:
        # Check environment variable first
        config_path = os.environ.get("CONFIG_PATH", str(base_dir / "configs" / "artemis.yaml"))
    
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = base_dir / config_file
        
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Parse sections
    db_cfg = DBConfig(
        url=os.environ.get("DATABASE_URL", raw.get("db", {}).get("url", ""))
    )
    
    router_cfg = RouterConfig(
        checkpoint_path=raw.get("router", {}).get("checkpoint_path", ""),
        config_file=raw.get("router", {}).get("config_file", ""),
        device=raw.get("router", {}).get("device", "cpu")
    )
    
    lb_cfg = LoadBalancerConfig(
        config_file=raw.get("load_balancer", {}).get("config_file", "")
    )
    
    # Load LB config file if specified to get models and SLA
    lb_config_path = base_dir / lb_cfg.config_file
    if lb_config_path.exists():
        with open(lb_config_path, 'r') as f:
            lb_raw = yaml.safe_load(f)
        lb_cfg.global_sla_ms = lb_raw.get("global", {}).get("latency_sla_ms", 2000)
        lb_cfg.models = lb_raw.get("models", {})
    
    inf_cfg = InferenceConfig(
        models_file=raw.get("inference_engine", {}).get("models_file", "")
    )
    
    dc_cfg = DataCollectionConfig(
        samples_table=raw.get("data_collection", {}).get("samples_table", "vlm_samples_collected"),
        responses_table=raw.get("data_collection", {}).get("responses_table", "vlm_responses_collected"),
        feedback_table=raw.get("data_collection", {}).get("feedback_table", "vlm_feedback")
    )
    
    rt_cfg = RetrainingConfig(
        epochs=raw.get("retraining", {}).get("epochs", 1),
        batch_size=raw.get("retraining", {}).get("batch_size", 8),
        output_checkpoint=raw.get("retraining", {}).get("output_checkpoint", "checkpoints/best_reward_router_updated.pt")
    )
    
    return GlobalConfig(
        db=db_cfg,
        router=router_cfg,
        load_balancer=lb_cfg,
        inference=inf_cfg,
        data_collection=dc_cfg,
        retraining=rt_cfg,
        _base_dir=str(base_dir)
    )

# Convenience accessors
def get_db_url(cfg: GlobalConfig) -> str:
    return cfg.db.url

def get_router_config(cfg: GlobalConfig) -> RouterConfig:
    return cfg.router

def get_load_balancer_config(cfg: GlobalConfig) -> LoadBalancerConfig:
    return cfg.load_balancer

def get_inference_config(cfg: GlobalConfig) -> InferenceConfig:
    return cfg.inference

def get_data_collection_config(cfg: GlobalConfig) -> DataCollectionConfig:
    return cfg.data_collection
