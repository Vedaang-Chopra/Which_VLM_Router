"""
Statistics registry for per-task/per-model performance metrics.

This module loads and provides access to statistics computed by the Ares module,
including latency, cost, and accuracy for each (task_type, model_name) combination.

Expected input format (JSON):
{
  "task_type": {
    "model_name": {
      "avg_latency_ms": 150.0,
      "avg_accuracy": 0.95,
      "cost_per_request_usd": 0.0005
    }
  }
}

Developers can refer to Ares notebooks under artemis_final/ares/ to see how
these statistics are computed from the dataset.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from .config import STATS_PATH

logger = logging.getLogger(__name__)


class StatsRegistry:
    """
    Registry for per-task/per-model statistics.
    
    Acts as the single source of truth for load balancer statistics.
    Supports in-memory updates and lazy loading from a JSON file.
    """

    def __init__(self, stats_dict: Optional[Dict] = None):
        """
        Initialize the stats registry.

        Args:
            stats_dict: Pre-loaded statistics dictionary. If None, will be loaded
                       from disk when needed via load_per_task_model_stats()
        """
        self._stats_dict = stats_dict
        self._missing_stats_warned = set()  # Track (task, model, stat_type) tuples we've warned about

    @property
    def stats_dict(self) -> Dict:
        """Lazy-load stats dict if not already loaded."""
        if self._stats_dict is None:
            self._stats_dict = load_per_task_model_stats()
        return self._stats_dict

    def get_stats_for(self, task_type: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific task and model.

        Args:
            task_type: Type of task (e.g., "ocr", "chart_vqa")
            model_name: Name of the model (e.g., "small_vlm")

        Returns:
            Dictionary with stats or None if not found
        """
        return self.stats_dict.get(task_type, {}).get(model_name)

    def estimate_service_time_ms(
        self,
        task_type: str,
        model_name: str,
        default_ms: float = 1000.0
    ) -> float:
        """
        Estimate service time for a task/model combination.

        Args:
            task_type: Type of task
            model_name: Name of the model
            default_ms: Default latency to use if stats not found (1000.0 ms)

        Returns:
            Estimated service time in milliseconds
        """
        stats = self.get_stats_for(task_type, model_name)
        if stats and "avg_latency_ms" in stats:
            return float(stats["avg_latency_ms"])
        
        self._warn_missing_stats(task_type, model_name, "avg_latency_ms")
        return default_ms

    def estimate_cost_usd(self, task_type: str, model_name: str) -> float:
        """
        Estimate cost for a task/model combination.

        Args:
            task_type: Type of task
            model_name: Name of the model

        Returns:
            Estimated cost in USD (default 0.0)
        """
        stats = self.get_stats_for(task_type, model_name)
        if stats and "cost_per_request_usd" in stats:
            return float(stats["cost_per_request_usd"])
            
        self._warn_missing_stats(task_type, model_name, "cost_per_request_usd")
        return 0.0

    def estimate_accuracy(self, task_type: str, model_name: str) -> float:
        """
        Estimate accuracy for a task/model combination.

        Args:
            task_type: Type of task
            model_name: Name of the model

        Returns:
            Estimated accuracy (0.0 to 1.0, default 0.0)
        """
        stats = self.get_stats_for(task_type, model_name)
        if stats and "avg_accuracy" in stats:
            return float(stats["avg_accuracy"])

        self._warn_missing_stats(task_type, model_name, "avg_accuracy")
        return 0.0

    def has_stats(self, task_type: str, model_name: str) -> bool:
        """
        Check if statistics are available for a task/model combination.

        Args:
            task_type: Type of task
            model_name: Name of the model

        Returns:
            True if stats are available, False otherwise
        """
        return self.get_stats_for(task_type, model_name) is not None

    def update_latency(self, task_type: str, model_name: str, latency_ms: float):
        """
        Update the latency statistic for a task and model.

        Args:
            task_type: Type of task
            model_name: Name of the model
            latency_ms: Latency in milliseconds
        """
        entry = self._ensure_entry(task_type, model_name)
        entry["avg_latency_ms"] = float(latency_ms)

    def update_accuracy(self, task_type: str, model_name: str, accuracy: float):
        """
        Update the accuracy statistic for a task and model.

        Args:
            task_type: Type of task
            model_name: Name of the model
            accuracy: Accuracy (0.0 to 1.0)
        """
        entry = self._ensure_entry(task_type, model_name)
        entry["avg_accuracy"] = float(accuracy)

    def update_cost(self, task_type: str, model_name: str, cost_usd: float):
        """
        Update the cost statistic for a task and model.

        Args:
            task_type: Type of task
            model_name: Name of the model
            cost_usd: Cost per request in USD
        """
        entry = self._ensure_entry(task_type, model_name)
        entry["cost_per_request_usd"] = float(cost_usd)

    def _ensure_entry(self, task_type: str, model_name: str) -> Dict[str, Any]:
        """
        Ensure a stats entry exists for the given task and model.
        
        Args:
            task_type: Type of task
            model_name: Name of the model
            
        Returns:
            The mutable dictionary for the specific task/model stats.
        """
        if self._stats_dict is None:
            # Initialize with empty dict if not loaded or non-existent
            self._stats_dict = {}
            
        if task_type not in self._stats_dict:
            self._stats_dict[task_type] = {}
            
        if model_name not in self._stats_dict[task_type]:
            self._stats_dict[task_type][model_name] = {}
            
        return self._stats_dict[task_type][model_name]

    def _warn_missing_stats(self, task_type: str, model_name: str, stat_type: str):
        """
        Log a warning for missing statistics (once per task/model/field combination).

        Args:
            task_type: Type of task
            model_name: Name of the model
            stat_type: Type of statistic being requested (e.g. 'avg_latency_ms')
        """
        key = (task_type, model_name, stat_type)
        if key not in self._missing_stats_warned:
            logger.warning(
                f"Missing '{stat_type}' stats for task='{task_type}', model='{model_name}'. "
                f"Using default value."
            )
            self._missing_stats_warned.add(key)


def load_per_task_model_stats(stats_path: Optional[Path] = None) -> Dict:
    """
    Load per-task/per-model statistics from JSON file.

    This file is produced by Ares notebooks under artemis_final/ares/.
    Developers can refer to those notebooks to understand how the statistics
    are computed.

    Args:
        stats_path: Path to the stats JSON file. If None, uses default path
                   from config (artemis_final/ares/aggregates/per_task_model_stats.json)

    Returns:
        Dictionary with structure: {task_type: {model_name: stats}}

    Raises:
        FileNotFoundError: If stats file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if stats_path is None:
        stats_path = STATS_PATH

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found at {stats_path}. "
            f"Please run Ares notebooks to generate per_task_model_stats.json"
        )

    with open(stats_path, 'r') as f:
        stats_dict = json.load(f)

    logger.info(f"Loaded stats for {len(stats_dict)} task types from {stats_path}")

    # Log summary
    total_combinations = sum(len(models) for models in stats_dict.values())
    logger.info(f"Total task/model combinations: {total_combinations}")

    return stats_dict


# Module-level convenience functions for backward compatibility
_default_registry: Optional[StatsRegistry] = None


def get_default_registry() -> StatsRegistry:
    """Get or create the default stats registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = StatsRegistry()
    return _default_registry


def get_stats_for(task_type: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Convenience function using default registry."""
    return get_default_registry().get_stats_for(task_type, model_name)


def estimate_service_time_ms(
    task_type: str,
    model_name: str,
    stats_dict: Optional[Dict] = None,
    default_ms: float = 1000.0
) -> float:
    """
    Convenience function to estimate service time.

    Args:
        task_type: Type of task
        model_name: Name of the model
        stats_dict: Optional stats dictionary. If None, uses default registry
        default_ms: Default latency if stats not found

    Returns:
        Estimated service time in milliseconds
    """
    if stats_dict is not None:
        registry = StatsRegistry(stats_dict)
    else:
        registry = get_default_registry()

    return registry.estimate_service_time_ms(task_type, model_name, default_ms)


def estimate_cost_usd(
    task_type: str,
    model_name: str,
    stats_dict: Optional[Dict] = None
) -> float:
    """
    Convenience function to estimate cost.

    Args:
        task_type: Type of task
        model_name: Name of the model
        stats_dict: Optional stats dictionary. If None, uses default registry

    Returns:
        Estimated cost in USD
    """
    if stats_dict is not None:
        registry = StatsRegistry(stats_dict)
    else:
        registry = get_default_registry()

    return registry.estimate_cost_usd(task_type, model_name)


def estimate_accuracy(
    task_type: str,
    model_name: str,
    stats_dict: Optional[Dict] = None
) -> float:
    """
    Convenience function to estimate accuracy.

    Args:
        task_type: Type of task
        model_name: Name of the model
        stats_dict: Optional stats dictionary. If None, uses default registry

    Returns:
        Estimated accuracy (0.0 to 1.0)
    """
    if stats_dict is not None:
        registry = StatsRegistry(stats_dict)
    else:
        registry = get_default_registry()

    return registry.estimate_accuracy(task_type, model_name)
