"""
LoadBalancerService: Wrapper around the Artemis load balancer.
Uses common.config_loader for configuration.
"""
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import asdict

# Import existing load balancer logic
try:
    from .scheduler import ArtemisLoadBalancer
    from .stats_registry import StatsRegistry
    from .config import ModelCapacityConfig, AutoscaleConfig
    from .types import RouterOutput, SchedulingContext
except ImportError:
    from load_balancer.scheduler import ArtemisLoadBalancer
    from load_balancer.stats_registry import StatsRegistry
    from load_balancer.config import ModelCapacityConfig, AutoscaleConfig
    from load_balancer.types import RouterOutput, SchedulingContext

from common.config_loader import GlobalConfig

logger = logging.getLogger(__name__)

class LoadBalancerService:
    """
    Service wrapper for the Artemis load balancer.
    Combines router decisions with SLA and load considerations.
    """
    
    def __init__(self, cfg: GlobalConfig):
        """
        Initialize the Load Balancer Service.
        
        Args:
            cfg: GlobalConfig from common.config_loader
        """
        self.cfg = cfg
        self.lb_cfg = cfg.load_balancer
        self.stats_registry = StatsRegistry()
        self.balancer = self._initialize_balancer()
        
        # Stats for record_outcome
        self._outcome_stats = {
            "total_requests": 0,
            "total_latency_ms": 0.0,
            "success_count": 0,
            "failure_count": 0
        }
        
    def _initialize_balancer(self) -> ArtemisLoadBalancer:
        """Initialize the Artemis scheduler."""
        logger.info("Initializing LoadBalancerService...")
        
        # Convert config dicts to ModelCapacityConfig objects
        model_configs = {}
        for model_name, model_data in self.lb_cfg.models.items():
            autoscale = None
            if 'autoscale' in model_data:
                ad = model_data['autoscale']
                autoscale = AutoscaleConfig(
                    enable=ad.get('enable', True),
                    scale_up_latency_factor=ad.get('scale_up_latency_factor', 0.8),
                    scale_down_util_threshold=ad.get('scale_down_util_threshold', 0.3),
                    cooldown_ms=ad.get('cooldown_ms', 60000)
                )
                
            model_configs[model_name] = ModelCapacityConfig(
                model_name=model_name,
                base_latency_ms=model_data.get('base_latency_ms', 1000.0),
                min_replicas=model_data.get('min_replicas', 1),
                max_replicas=model_data.get('max_replicas', 1),
                sla_ms=model_data.get('sla_ms', 2000.0),
                max_qps_per_replica=model_data.get('max_qps_per_replica', 1.0),
                cost_per_request_usd=model_data.get('cost_per_request_usd', 0.0001),
                autoscale=autoscale
            )
            
        return ArtemisLoadBalancer(
            model_configs=model_configs,
            stats_registry=self.stats_registry,
            global_latency_sla_ms=self.lb_cfg.global_sla_ms,
            scheduling_mode="capacity_aware"
        )

    def schedule(self, 
                 sample_id: str,
                 task_type: str,
                 router_probs: Dict[str, float],
                 preferred_model: str,
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Schedule a request based on router decision and system state.
        
        Args:
            sample_id: Unique request ID
            task_type: Type of task (e.g., "vlm")
            router_probs: Model probability scores from router
            preferred_model: Router's preferred model
            metadata: Additional metadata
            
        Returns:
            Dict containing scheduling decision
        """
        router_output = RouterOutput(
            sample_id=sample_id,
            task_type=task_type,
            router_probs=router_probs,
            preferred_model=preferred_model
        )
        
        context = SchedulingContext(
            arrival_ts_ms=time.time() * 1000,
            load_profile="production",
            metadata=metadata or {}
        )
        
        decision = self.balancer.schedule(router_output, context)
        return asdict(decision)

    def record_outcome(self, lb_decision: Any, outcome: Dict[str, Any]):
        """
        Record the outcome of a request for stats tracking.
        
        Args:
            lb_decision: The LBDecision that was made
            outcome: Dict with keys like latency_ms, success, error
        """
        self._outcome_stats["total_requests"] += 1
        
        latency = outcome.get("latency_ms", 0)
        self._outcome_stats["total_latency_ms"] += latency
        
        if outcome.get("success", True):
            self._outcome_stats["success_count"] += 1
        else:
            self._outcome_stats["failure_count"] += 1
        
        # Log periodically
        if self._outcome_stats["total_requests"] % 100 == 0:
            avg_latency = self._outcome_stats["total_latency_ms"] / self._outcome_stats["total_requests"]
            success_rate = self._outcome_stats["success_count"] / self._outcome_stats["total_requests"]
            logger.info(f"LB Stats: {self._outcome_stats['total_requests']} requests, "
                       f"avg_latency={avg_latency:.1f}ms, success_rate={success_rate:.2%}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current load balancer statistics."""
        return self._outcome_stats.copy()
