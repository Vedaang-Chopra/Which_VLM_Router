from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class ModelEndpoint:
    name: str # Logical name (e.g. gemma_3_27b)
    base_url: str
    model_id: str # HuggingFace ID (e.g. google/gemma-3-27b-it)
    api_key: str = "EMPTY"
    pricing: Dict[str, float] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    # Simple defaults for Load Balancer simulation
    base_latency_ms: float = 1000.0
    sla_ms: float = 3000.0
    max_qps_per_replica: float = 1.0

def load_endpoints_from_config(config_models: List[Dict[str, Any]]) -> List[ModelEndpoint]:
    endpoints = []
    for m in config_models:
        eps = ModelEndpoint(
            name=m['name'],
            base_url=m['base_url'].rstrip('/'),
            model_id=m['model_id'],
            api_key=m.get('api_key', 'EMPTY'),
            pricing=m.get('pricing', {}),
            extra_params=m.get('extra_params', {}),
            base_latency_ms=m.get('base_latency_ms', 1000.0),
            sla_ms=m.get('sla_ms', 3000.0),
            max_qps_per_replica=m.get('max_qps_per_replica', 1.0)
        )
        endpoints.append(eps)
    return endpoints
