"""
LoadBalancerService: Wrapper around the Artemis load balancer.
Uses common.config_loader for configuration, but delegates core logic to the public API module.
"""
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from common.config_loader import GlobalConfig

# Import from public API for standardized access
from .public_api import ArtemisLoadBalancerModule, init_load_balancer, schedule_request, get_metrics

logger = logging.getLogger(__name__)

class LoadBalancerService:
    """
    Service wrapper for the Artemis load balancer.
    Combines router decisions with SLA and load considerations.
    
    This class adapts the global public API to the application's service structure.
    """
    
    def __init__(self, cfg: GlobalConfig):
        """
        Initialize the Load Balancer Service.
        
        Args:
            cfg: GlobalConfig from common.config_loader
        """
        self.cfg = cfg
        self.lb_cfg = cfg.load_balancer
        
        # Initialize the underlying module
        # Note: In a real production setup, we might map 'cfg' to the specific config structure
        # the module expects, or allow the module to load its own config. 
        # For now, we rely on the module's default loading mechanism or pass explicit paths if needed.
        # If the cfg object has specific overrides, we should pass them. 
        
        # Initialize the global instance primarily
        # We can also instantiate a local module if we want isolation, but `init_load_balancer` sets a global.
        # Let's use a local instance to be safe and encapsulated within this service class.
        
        self.module = ArtemisLoadBalancerModule() 
        # TODO: If we want to fully respect 'cfg' overrides (like specific model configs passed in memory),
        # we would need to update ArtemisLoadBalancerModule to accept them.
        # Current implementation of ArtemisLoadBalancerModule loads from disk or defaults.
        
        # Stats for record_outcome
        self._outcome_stats = {
            "total_requests": 0,
            "total_latency_ms": 0.0,
            "success_count": 0,
            "failure_count": 0
        }
        logger.info("LoadBalancerService initialized via Public API Module.")

    def schedule(self, 
                 sample_id: str,
                 task_type: str,
                 router_probs: Dict[str, float],
                 preferred_model: str,
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Schedule a request based on router decision and system state.
        Delegates to the internal logic module.
        """
        return self.module.schedule(
            sample_id=sample_id,
            task_type=task_type,
            router_probs=router_probs,
            preferred_model=preferred_model,
            metadata=metadata
        )

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
        # Mix outcome stats with internal SLA stats
        stats = self._outcome_stats.copy()
        stats.update(self.module.get_sla_summary())
        return stats
