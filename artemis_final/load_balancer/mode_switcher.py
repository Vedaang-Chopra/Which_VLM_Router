from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModeSwitchConfig:
    default_mode: str = "balanced"
    violation_threshold: float = 0.10  # 10% SLA violation rate threshold
    scaled_cooldown_sec: int = 300     # 5 minutes cooldown between switches
    
class ModeSwitcher:
    """Handles runtime mode switching based on SLA violations and system state."""

    def __init__(self, config: ModeSwitchConfig):
        self.config = config
        self.violation_threshold = config.violation_threshold
        self.cooldown_sec = config.scaled_cooldown_sec
        self.last_switch_time = 0
        self.current_mode = config.default_mode
        self.switch_history = []

    def should_switch_mode(self, sla_stats: Any) -> str:
        """
        Determine if mode switch is needed based on violations.
        Returns the mode to use (either new or current).
        
        sla_stats expected to have:
        - global_accuracy
        - latency_violation_rate (0.0-1.0)
        - budget_remaining_pct (0.0-1.0)
        - min_global_accuracy (config)
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_switch_time < self.cooldown_sec:
            return self.current_mode

        target_mode = self.current_mode

        # 1. Critical: Accuracy Safety Net
        # If accuracy drops below minimum, force accuracy mode
        if hasattr(sla_stats, 'global_accuracy') and hasattr(sla_stats, 'min_global_accuracy'):
            if sla_stats.global_accuracy < sla_stats.min_global_accuracy:
                target_mode = "accuracy"

        # 2. Performance: High Latency Violations
        # If we are failing SLAs too often, switch to speed
        # But only if we aren't already forced into accuracy mode (accuracy takes precedence here? 
        # Actually, if we miss SLAs, accuracy doesn't matter if response is too late. 
        # But usually 'accuracy' mode is slowest. 'fast' is fastest.
        # Let's assess priority. Typically: Budget > Accuracy > Latency or Budget > Latency > Accuracy?
        # User prompt implies:
        # Accuracy < threshold -> accuracy
        # Latency > threshold -> fast
        # Budget < threshold -> cheap
        
        # Let's check budget first as it's a hard constraint often.
        if hasattr(sla_stats, 'budget_remaining_pct') and sla_stats.budget_remaining_pct < 0.2:
             target_mode = "cheap"
        
        # Check accuracy next 
        elif hasattr(sla_stats, 'global_accuracy') and hasattr(sla_stats, 'min_global_accuracy') and \
             sla_stats.global_accuracy < sla_stats.min_global_accuracy:
             target_mode = "accuracy"
             
        # Check latency
        elif hasattr(sla_stats, 'latency_violation_rate') and \
             sla_stats.latency_violation_rate > self.violation_threshold:
             target_mode = "fast"
        
        # If all good, revert to default/balanced? 
        # "Otherwise, stay in balanced" per prompt.
        else:
            target_mode = self.config.default_mode

        # Execute switch if changed
        if target_mode != self.current_mode:
            logger.info(f"Switching mode: {self.current_mode} -> {target_mode}")
            self.current_mode = target_mode
            self.last_switch_time = current_time
            self.switch_history.append((current_time, target_mode))
            
        return self.current_mode
