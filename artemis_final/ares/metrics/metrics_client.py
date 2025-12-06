"""Client for fetching GPU metrics from metrics server."""

import requests
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GPUMetricsClient:
    """Fetch GPU metrics from FastAPI metrics servers.
    
    Supports multi-GPU nodes where each VLM is pinned to a specific GPU.
    Use gpu_indices to map model_name -> gpu_index for filtering.
    """
    
    def __init__(
        self, 
        endpoints: Dict[str, str] = None, 
        gpu_indices: Dict[str, int] = None,
        timeout: float = 15.0
    ):
        """
        Args:
            endpoints: Dict mapping model_name -> metrics URL
            gpu_indices: Dict mapping model_name -> GPU index (0, 1, etc.)
            timeout: Request timeout in seconds
        """
        self.endpoints = endpoints or {}
        self.gpu_indices = gpu_indices or {}
        self.timeout = timeout
        self._cache: Dict[str, Dict] = {}  # URL -> metrics cache
    
    def fetch_metrics(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Fetch metrics for a specific model's server."""
        url = self.endpoints.get(model_name)
        if not url:
            return None
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch metrics for {model_name}: {e}")
            return None
    
    def fetch_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Fetch metrics for all configured models."""
        return {name: self.fetch_metrics(name) for name in self.endpoints}
    
    def get_gpu_summary(
        self, 
        model_name: str, 
        gpu_index: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get GPU summary for a specific model, filtered by gpu_index.
        
        Args:
            model_name: Name of the model
            gpu_index: Override GPU index (uses self.gpu_indices[model_name] if not provided)
            
        Returns:
            Dict with GPU metrics, or None if unavailable
        """
        metrics = self.fetch_metrics(model_name)
        if not metrics or 'gpu' not in metrics:
            return None
        
        gpus = metrics['gpu'].get('gpus', [])
        if not gpus:
            return None
        
        # Determine which GPU index to use
        idx = gpu_index if gpu_index is not None else self.gpu_indices.get(model_name, 0)
        
        # Validate index
        if idx >= len(gpus):
            logger.warning(f"GPU index {idx} out of range for {model_name} (only {len(gpus)} GPUs available)")
            return None
        
        gpu = gpus[idx]
        
        # Extract comprehensive GPU metrics
        memory = gpu.get('memory', {})
        utilization = gpu.get('utilization', {})
        
        return {
            'gpu_index': idx,
            'gpu_name': gpu.get('name'),
            'util_percent': utilization.get('gpu_percent'),
            'memory_util_percent': utilization.get('memory_percent'),
            'mem_used_mb': memory.get('used_mb'),
            'mem_total_mb': memory.get('total_mb'),
            'mem_free_mb': memory.get('free_mb'),
            'temp_celsius': gpu.get('temperature_celsius'),
            'power_watts': gpu.get('power_watts'),
            'power_limit_watts': gpu.get('power_limit_watts'),
        }
    
    def get_full_metrics_for_model(
        self, 
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get full metrics including system info for a model.
        
        Returns the complete metrics response filtered to the model's GPU.
        """
        metrics = self.fetch_metrics(model_name)
        if not metrics:
            return None
        
        gpu_summary = self.get_gpu_summary(model_name)
        
        return {
            'timestamp_utc': metrics.get('timestamp_utc'),
            'gpu': gpu_summary,
            'cpu_usage_percent': metrics.get('cpu', {}).get('cpu_usage_percent'),
            'memory_used_percent': metrics.get('memory', {}).get('virtual_memory', {}).get('percent'),
        }
