"""Client for fetching GPU metrics from metrics server."""

import requests
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GPUMetricsClient:
    """Fetch GPU metrics from FastAPI metrics servers."""
    
    def __init__(self, endpoints: Dict[str, str] = None, timeout: float = 5.0):
        """
        Args:
            endpoints: Dict mapping model_name -> metrics URL
            timeout: Request timeout in seconds
        """
        self.endpoints = endpoints or {}
        self.timeout = timeout
    
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
    
    def get_gpu_summary(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get simplified GPU summary."""
        metrics = self.fetch_metrics(model_name)
        if not metrics or 'gpu' not in metrics:
            return None
        
        gpus = metrics['gpu'].get('gpus', [])
        if not gpus:
            return None
        
        gpu = gpus[0]
        return {
            'util_percent': gpu.get('utilization', {}).get('gpu_percent'),
            'mem_used_mb': gpu.get('memory', {}).get('used_mb'),
            'mem_total_mb': gpu.get('memory', {}).get('total_mb'),
            'temp_celsius': gpu.get('temperature_celsius'),
            'power_watts': gpu.get('power_watts'),
        }