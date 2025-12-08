"""
Public API Facade for Artemis Load Balancer.

This module provides a simplified, high-level interface to the Artemis load balancer,
hiding the implementation details of the scheduler, stats registry, and SLA monitors.

Usage:
    from artemis_final.load_balancer.public_api import init_load_balancer, schedule_request, get_metrics

    # 1. Initialize (loads config and stats)
    init_load_balancer()

    # 2. Schedule a request
    decision = schedule_request(
        sample_id="test_001",
        task_type="vqa",
        router_probs={"qwen2_5_vl_7b": 0.8, "gemma_3_27b": 0.2},
        preferred_model="qwen2_5_vl_7b"
    )
    print(f"Chosen model: {decision['chosen_model']}")

    # 3. Check metrics
    metrics = get_metrics()
    print(f"Avg Latency: {metrics['latency_p50_ms']} ms")
"""

import time
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
from dataclasses import asdict

from .core.config import (
    load_capacity_config,
    default_experiment_config,
    CAPACITY_CONFIG_PATH,
    STATS_PATH,
    ModelCapacityConfig,
    GlobalSLAConfig
)
from .core.stats_registry import StatsRegistry, load_per_task_model_stats
from .core.scheduler import ArtemisLoadBalancer
from .core.sla_monitor import SlaMonitor
from .core.types import RouterOutput, SchedulingContext, SchedulingDecision, BudgetExhaustedError
from .core.model_state import ModelStateManager

logger = logging.getLogger(__name__)

class ArtemisLoadBalancerModule:
    """
    High-level wrapper around the Artemis load balancer.

    This class serves as the central entry point for load balancing operations, encapsulating
    configuration loading, stats management, request scheduling, and SLA monitoring.

    Key responsibilities:
    - **Initialization**: Automatically loads capacity configurations and historical statistics.
    - **Scheduling**: Determines the optimal model for a given request based on current load, costs, and router probabilities.
    - **Monitoring**: Tracks system performance against Service Level Agreements (SLAs).
    - **Simulation**: Provides utilities for running synthetic traffic to validate configurations.

    Usage Example:
        lb = ArtemisLoadBalancerModule()
        decision = lb.schedule(
            sample_id="req_123",
            task_type="vqa",
            router_probs={"model_a": 0.9, "model_b": 0.1},
            preferred_model="model_a"
        )
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        stats_dict: Optional[dict] = None
    ):
        """
        Initialize the load balancer module.

        Args:
            config_path: Path to capacity_config.yaml. If None, uses default.
            stats_dict: Pre-loaded stats dictionary. If None, loads from default JSON path.
        """
        # 1. Load Configuration
        self.config_path = Path(config_path) if config_path else CAPACITY_CONFIG_PATH
        
        # Load model capacity configs
        # If config_path doesn't exist and no internal defaults, we warn.
        # But we also want to support passing config objects directly maybe?
        # For this simplified API, we stick to path or defaults.
        if not self.config_path.exists():
             logger.warning(f"Config path {self.config_path} not found. using defaults.")
        
        self.model_configs = load_capacity_config(self.config_path)
        
        # Load Experiment/Global Config (for SLAs)
        # Note: We use default_experiment_config as a base for global settings
        # In a real app, this might come from a separate global config file.
        self.exp_config = default_experiment_config()
        self.latency_slas = self.exp_config.latency_sla_ms
        self.global_sla = self.exp_config.global_sla
        
        # 2. Initialize Stats Registry
        if stats_dict is None:
            # Try to load from default path if exists, otherwise empty
            if STATS_PATH.exists():
                stats_dict = load_per_task_model_stats(STATS_PATH)
            else:
                logger.warning(f"Stats file {STATS_PATH} not found. Starting with empty stats.")
                stats_dict = {}
        
        self.stats_registry = StatsRegistry(stats_dict)

        # 3. Initialize Scheduler
        # We default to 'capacity_aware' mode as a safe general default
        self.scheduler = ArtemisLoadBalancer(
            model_configs=self.model_configs,
            stats_registry=self.stats_registry,
            latency_sla_ms=self.latency_slas,
            global_sla_config=self.global_sla,
            scheduling_mode="capacity_aware", 
            simulation_only=False # Online mode by default
        )
        
        # 4. Initialize Monitor
        # Tracks metrics for the lifetime of this module instance
        self.sla_monitor = SlaMonitor(
            latency_sla_ms=self.global_sla.default_latency_ms
        )
        
        logger.info("ArtemisLoadBalancerModule initialized successfully.")

    def schedule(
        self,
        sample_id: str,
        task_type: str,
        router_probs: Dict[str, float],
        preferred_model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Schedule a single request.

        Args:
            sample_id: Unique ID for the request.
            task_type: Task type (e.g. 'vqa', 'ocr').
            router_probs: Probability dict {model_name: probability}.
            preferred_model: The model the router prefers (highest prob).
            metadata: Optional extra metadata for logging.

        Returns:
            Dict[str, Any]: A dictionary containing the scheduling decision details.
            
            Key fields include:
            - `chosen_model` (str): The name of the model selected to handle the request.
            - `is_overloaded` (bool): True if the chosen model (or system) is currently earning high load.
            - `estimated_latency_ms` (float): The expected latency for this request.
            - `estimated_cost` (float): The estimated cost for processing this request.
            - `modified_probs` (Dict[str, float]): The adjusted probabilities used for making the decision (if applicable).
            - `routing_method` (str): The strategy used (e.g., 'optimization', 'fallback').
        """
        # 1. Prepare Inputs
        router_output = RouterOutput(
            sample_id=sample_id,
            task_type=task_type,
            router_probs=router_probs,
            preferred_model=preferred_model,
            max_prob=router_probs.get(preferred_model, 0.0)
        )
        
        context = SchedulingContext(
            arrival_ts_ms=time.time() * 1000,
            load_profile="production", # Default tag
            metadata=metadata or {}
        )

        # 2. Schedule
        decision: SchedulingDecision = self.scheduler.schedule(router_output, context)

        # 3. Update Monitor 
        # (Internal scheduler monitor updates automatically, but we keep our own 
        # separate monitor if we want isolation, or rely on scheduler's. 
        # Here we rely on scheduler's internal SlaMonitor for consistency, 
        # and also update our own if needed. 
        # Actually, ArtemisLoadBalancer has its own SlaMonitor. 
        # Let's wrap that one instead of maintaining a duplicate.)
        
        # For this facade, we can just return the dict.
        return asdict(decision)

    def get_sla_summary(self) -> Dict[str, Any]:
        """
        Return a dictionary with basic SLA metrics from the internal monitor.
        """
        # Access the monitor inside the scheduler
        metrics = self.scheduler.sla_monitor.snapshot()
        return asdict(metrics)

    def reset_metrics(self) -> None:
        """
        Clear all accumulated decisions/metrics.
        """
        self.scheduler.sla_monitor.reset()
        # Also reset scheduler state if completely restarting
        self.scheduler.reset() 

    def run_synthetic_simulation(self, num_requests: int = 100) -> List[Dict[str, Any]]:
        """
        Run a tiny synthetic simulation using fake router outputs.
        
        Useful for sanity checking configuration without a real workload.
        """
        import random
        models = list(self.model_configs.keys())
        task_types = ["vqa", "ocr", "captioning"]
        
        results = []
        for i in range(num_requests):
            t_type = random.choice(task_types)
            
            # Fake probs
            probs = {m: random.random() for m in models}
            total = sum(probs.values())
            probs = {m: p/total for m, p in probs.items()}
            
            pref = max(probs, key=probs.get)
            
            decision = self.schedule(
                sample_id=f"synth_{i}",
                task_type=t_type,
                router_probs=probs,
                preferred_model=pref
            )
            results.append(decision)
            
        return results

# -----------------------------------------------------------------------------
# Singleton / Module-Level Interface
# -----------------------------------------------------------------------------

_GLOBAL_MODULE_INSTANCE: Optional[ArtemisLoadBalancerModule] = None

def init_load_balancer(config_path: Optional[str] = None, stats_dict: Optional[dict] = None) -> None:
    """
    Initialize the global load balancer instance.
    
    Args:
        config_path: Path to capacity_config.yaml
        stats_dict: Optional pre-loaded stats
    """
    global _GLOBAL_MODULE_INSTANCE
    _GLOBAL_MODULE_INSTANCE = ArtemisLoadBalancerModule(config_path, stats_dict)

def schedule_request(
    sample_id: str,
    task_type: str,
    router_probs: Dict[str, float],
    preferred_model: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Schedule a request using the global load balancer.
    
    Must call init_load_balancer() first.
    """
    assert _GLOBAL_MODULE_INSTANCE is not None, "Load balancer not initialized. Call init_load_balancer() first."
    return _GLOBAL_MODULE_INSTANCE.schedule(
        sample_id, task_type, router_probs, preferred_model, metadata
    )

def get_metrics() -> Dict[str, Any]:
    """
    Get current SLA metrics from the global load balancer.
    """
    assert _GLOBAL_MODULE_INSTANCE is not None, "Load balancer not initialized. Call init_load_balancer() first."
    return _GLOBAL_MODULE_INSTANCE.get_sla_summary()

def reset_load_balancer_metrics() -> None:
    """
    Reset metrics and state of the global load balancer.
    """
    assert _GLOBAL_MODULE_INSTANCE is not None, "Load balancer not initialized. Call init_load_balancer() first."
    _GLOBAL_MODULE_INSTANCE.reset_metrics()
