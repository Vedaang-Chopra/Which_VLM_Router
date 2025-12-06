"""
Artemis Router - Fast, modular router inference service for VLM routing.

This package provides:
- Low-latency router inference
- Multiple input sources (SQL, HTTP, synthetic)
- SQL and W&B logging
- Load balancer integration
- Traffic simulation utilities
"""

from .config import AppConfig, load_config
from .schemas import Sample, RouterDecision, InferenceResult, LogRecord
from .router_engine import RouterEngine

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "load_config",
    "Sample",
    "RouterDecision",
    "InferenceResult",
    "LogRecord",
    "RouterEngine",
]
