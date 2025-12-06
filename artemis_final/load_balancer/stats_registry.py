"""
Statistics registry for per-task/per-model performance metrics.

This module loads and provides access to statistics computed by the Ares module,
including latency, cost, and accuracy for each (task_type, model_name) combination.

Expected input format (JSON):
{
  "ocr": {
    "small_vlm": {
      "avg_latency_ms": 200.0,
      "avg_accuracy": 0.85,
      "cost_per_token_usd": 0.000002,
      "avg_tokens": 300
    },
    ...
  },
  ...
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

    This class loads statistics from the Ares aggregates and provides
    helper methods to retrieve stats for specific task/model combinations.
    """

    def __init__(self, stats_dict: Optional[Dict] = None):
        """
        Initialize the stats registry.

        Args:
            stats_dict: Pre-loaded statistics dictionary. If None, will be loaded
                       when needed via load_per_task_model_stats()
        """
        self._stats_dict = stats_dict
        self._missing_stats_warned = set()  # Track (task, model) pairs we've warned about

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
            default_ms: Default latency to use if stats not found

        Returns:
            Estimated service time in milliseconds
        """
        stats = self.get_stats_for(task_type, model_name)
        if stats is None:
            self._warn_missing_stats(task_type, model_name, "service_time")
            return default_ms

        return stats.get("avg_latency_ms", default_ms)

    def estimate_cost_usd(self, task_type: str, model_name: str) -> float:
        """
        Estimate cost for a task/model combination.

        Args:
            task_type: Type of task
            model_name: Name of the model

        Returns:
            Estimated cost in USD
        """
        stats = self.get_stats_for(task_type, model_name)
        if stats is None:
            self._warn_missing_stats(task_type, model_name, "cost")
            return 0.0

        cost_per_token = stats.get("cost_per_token_usd", 0.0)
        avg_tokens = stats.get("avg_tokens", 0)
        return cost_per_token * avg_tokens

    def estimate_accuracy(self, task_type: str, model_name: str) -> float:
        """
        Estimate accuracy for a task/model combination.

        Args:
            task_type: Type of task
            model_name: Name of the model

        Returns:
            Estimated accuracy (0.0 to 1.0)
        """
        stats = self.get_stats_for(task_type, model_name)
        if stats is None:
            self._warn_missing_stats(task_type, model_name, "accuracy")
            return 0.0

        return stats.get("avg_accuracy", 0.0)

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

    def _warn_missing_stats(self, task_type: str, model_name: str, stat_type: str):
        """
        Log a warning for missing statistics (once per task/model pair).

        Args:
            task_type: Type of task
            model_name: Name of the model
            stat_type: Type of statistic being requested
        """
        key = (task_type, model_name)
        if key not in self._missing_stats_warned:
            logger.warning(
                f"Missing {stat_type} stats for task={task_type}, model={model_name}. "
                f"Using default values. Check Ares aggregates to ensure stats are available."
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
