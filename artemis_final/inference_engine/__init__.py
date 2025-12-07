"""
Inference Engine

Production-hardened VLM/LLM inference client with:
- Retry logic with exponential backoff
- Logprobs extraction for confidence scoring
- DeepSeek-OCR special handling
- Graceful error handling

Usage:
    from inference_engine.client import WhichVLMClient
    
    client = WhichVLMClient.from_yaml("models.yaml")
    client.vlm.run_image("image.png", "Describe this image", models="all")
"""

from .client import WhichVLMClient
from .config import ModelEndpoint, load_models_from_yaml, load_models_from_json
from .runners import OpenAIStyleRunner
from .suites import LLMTestSuite, VLMTestSuite

__all__ = [
    "WhichVLMClient",
    "ModelEndpoint",
    "load_models_from_yaml",
    "load_models_from_json",
    "OpenAIStyleRunner",
    "LLMTestSuite",
    "VLMTestSuite",
]
