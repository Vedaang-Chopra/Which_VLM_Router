"""
Unified configuration loader for the Artemis VLM Router system.
Loads from configs/artemis.yaml and provides typed access to all config sections.
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class DBConfig:
    url: str

@dataclass
class RouterConfig:
    checkpoint_path: str
    config_file: str
    device: str = "cpu"

@dataclass
class GlobalSLAConfig:
    total_cost_budget_usd: float
    min_global_accuracy: float
    default_latency_ms: int

@dataclass
class LoadBalancerConfig:
    global_sla: GlobalSLAConfig
    task_slas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    max_accuracy_drop: float = 0.05
    default_scheduling_mode: str = "capacity_aware"

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
    data_collection: DataCollectionConfig
    retraining: RetrainingConfig
    models: List[Dict[str, Any]] = field(default_factory=list)
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
        # Default is now in the same directory as this file (common/artemis.yaml)
        config_path = os.environ.get("CONFIG_PATH", str(Path(__file__).parent / "artemis.yaml"))
    
    config_file = Path(config_path)
    if not config_file.is_absolute():
        # If relative, assume relative to project base or just resolve it
        # If it was constructed via Path(__file__).parent, it's absolute
        config_file = base_dir / config_file
        
    if not config_file.exists():
        # Fallback check: maybe it's relative to base_dir/common/
        fallback = base_dir / "common" / "artemis.yaml"
        if fallback.exists():
            config_file = fallback
            
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        raw = yaml.safe_load(f)
    
    # DB
    db_raw = raw.get("db", {})
    db_cfg = DBConfig(
        url=os.environ.get("DATABASE_URL", db_raw.get("url", ""))
    )
    
    # Router
    rt_raw = raw.get("router", {})
    router_cfg = RouterConfig(
        checkpoint_path=rt_raw.get("checkpoint_path", ""),
        config_file=rt_raw.get("config_file", ""),
        device=rt_raw.get("device", "cpu")
    )
    
    # Load Balancer
    lb_raw = raw.get("load_balancer", {})
    gsla_raw = lb_raw.get("global_sla", {})
    gsla_cfg = GlobalSLAConfig(
        total_cost_budget_usd=gsla_raw.get("total_cost_budget_usd", 10.0),
        min_global_accuracy=gsla_raw.get("min_global_accuracy", 0.85),
        default_latency_ms=gsla_raw.get("default_latency_ms", 2000)
    )
    
    lb_cfg = LoadBalancerConfig(
        global_sla=gsla_cfg,
        task_slas=lb_raw.get("task_slas", {}),
        max_accuracy_drop=lb_raw.get("max_accuracy_drop", 0.05),
        default_scheduling_mode=lb_raw.get("default_scheduling_mode", "capacity_aware")
    )

    # Models
    models_list = raw.get("models", [])
    
    # Data Collection
    dc_raw = raw.get("data_collection", {})
    dc_cfg = DataCollectionConfig(
        samples_table=dc_raw.get("samples_table", "vlm_samples_collected"),
        responses_table=dc_raw.get("responses_table", "vlm_responses_collected"),
        feedback_table=dc_raw.get("feedback_table", "vlm_feedback")
    )
    
    # Retraining
    retr_raw = raw.get("retraining", {})
    retr_cfg = RetrainingConfig(
        epochs=retr_raw.get("epochs", 1),
        batch_size=retr_raw.get("batch_size", 8),
        output_checkpoint=retr_raw.get("output_checkpoint", "checkpoints/best_reward_router_updated.pt")
    )
    
    return GlobalConfig(
        db=db_cfg,
        router=router_cfg,
        load_balancer=lb_cfg,
        data_collection=dc_cfg,
        retraining=retr_cfg,
        models=models_list,
        _base_dir=str(base_dir)
    )

# Convenience accessors
def get_db_url(cfg: GlobalConfig) -> str:
    return cfg.db.url

def get_router_config(cfg: GlobalConfig) -> RouterConfig:
    return cfg.router

def get_load_balancer_config(cfg: GlobalConfig) -> LoadBalancerConfig:
    return cfg.load_balancer

def get_models_config(cfg: GlobalConfig) -> List[Dict[str, Any]]:
    return cfg.models

def get_data_collection_config(cfg: GlobalConfig) -> DataCollectionConfig:
    return cfg.data_collection
