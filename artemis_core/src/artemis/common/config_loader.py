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
    task_aggregates_path: str = ""

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
class GlobalConfig:
    db: DBConfig
    router: RouterConfig
    load_balancer: LoadBalancerConfig
    data_collection: DataCollectionConfig
    models: List[Dict[str, Any]] = field(default_factory=list)

def load_config(config_path: Optional[str] = None) -> GlobalConfig:
    if not config_path:
        # Default to ../../../config/artemis.yaml relative to this file
        config_path = str(Path(__file__).resolve().parents[3] / "config" / "artemis.yaml")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Configuration file is empty or invalid: {config_path}")

    # Helper to enforce required keys
    def require(d: Dict, key: str, parent: str):
        if key not in d:
            raise ValueError(f"Missing required configuration key: '{parent}.{key}'")

    require(raw, "db", "root")
    require(raw, "router", "root")
    require(raw, "load_balancer", "root")

    db_raw = raw.get("db", {})
    require(db_raw, "url", "db")

    router_raw = raw.get("router", {})
    require(router_raw, "checkpoint_path", "router")

    # Validate load_balancer.global_sla exists
    lb_raw = raw.get("load_balancer", {})
    require(lb_raw, "global_sla", "load_balancer")

    global_sla_raw = lb_raw.get("global_sla", {})
    require(global_sla_raw, "total_cost_budget_usd", "load_balancer.global_sla")
    require(global_sla_raw, "min_global_accuracy", "load_balancer.global_sla")
    require(global_sla_raw, "default_latency_ms", "load_balancer.global_sla")

    # Validate data_collection fields
    dc_raw = raw.get("data_collection", {})
    require(dc_raw, "samples_table", "data_collection")
    require(dc_raw, "responses_table", "data_collection")
    require(dc_raw, "feedback_table", "data_collection")

    # Construct config objects strictly
    return GlobalConfig(
        db=DBConfig(url=db_raw["url"]),
        router=RouterConfig(
            checkpoint_path=router_raw["checkpoint_path"],
            config_file=router_raw.get("config_file", ""),
            device=router_raw.get("device", "cpu"),
            task_aggregates_path=router_raw.get("task_aggregates_path", "")
        ),
        load_balancer=LoadBalancerConfig(
            global_sla=GlobalSLAConfig(**global_sla_raw),
            task_slas=lb_raw.get("task_slas", {}),
            max_accuracy_drop=lb_raw.get("max_accuracy_drop", 0.05),
            default_scheduling_mode=lb_raw.get("default_scheduling_mode", "capacity_aware")
        ),
        data_collection=DataCollectionConfig(**dc_raw),
        models=raw.get("models", [])
    )
