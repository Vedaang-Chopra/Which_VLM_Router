"""
Artemis Router - Fast, modular router inference service for VLM routing.

This package provides:
- Low-latency router inference via 3 router types
- Confidence-based fallback for uncertain decisions
- Data schemas for samples and routing decisions
- Load balancer integration
- Traffic simulation utilities
"""

from .schemas import Sample, RouterDecision, InferenceResult, LogRecord
from .legacy.inference_classical_router import ClassicalRouterInference
from .legacy.inference_pairwise_router import PairwiseRouterInference
from .inference_reward_router import RewardRouterInference
from .fallback import RouterFallback, FallbackConfig, FallbackResult, create_fallback_handler

__version__ = "0.1.0"

__all__ = [
    # Router inference classes
    "ClassicalRouterInference",
    "PairwiseRouterInference",
    "RewardRouterInference",
    # Fallback handling
    "RouterFallback",
    "FallbackConfig",
    "FallbackResult",
    "create_fallback_handler",
    # Data schemas
    "Sample",
    "RouterDecision",
    "InferenceResult",
    "LogRecord",
]
