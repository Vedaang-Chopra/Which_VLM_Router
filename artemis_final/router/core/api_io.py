"""
API I/O types for HTTP-style router requests.

Prepares for future FastAPI integration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
from PIL import Image
import requests
from io import BytesIO

from .schemas import Sample, InferenceResult


@dataclass
class RouteRequest:
    """
    HTTP API request format for routing.

    Attributes:
        text: Question/prompt text
        image_url: Optional URL to image
        image_data: Optional base64-encoded image data
        metadata: Additional metadata
    """
    text: str
    image_url: Optional[str] = None
    image_data: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RouteResponse:
    """
    HTTP API response format for routing.

    Attributes:
        sample_id: Sample identifier
        chosen_model: Selected model
        probs: Probability distribution
        router_inference_ms: Router latency
    """
    sample_id: str
    chosen_model: str
    probs: Dict[str, float]
    router_inference_ms: float


def sample_from_route_request(req: RouteRequest, sample_id: str) -> Sample:
    """
    Convert RouteRequest to Sample object.

    Args:
        req: HTTP request
        sample_id: ID to assign to sample

    Returns:
        Sample object
    """
    image = None
    image_uri = None

    # Load image from URL
    if req.image_url:
        try:
            response = requests.get(req.image_url, timeout=5)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_uri = req.image_url
        except Exception as e:
            print(f"[WARN] Failed to load image from URL {req.image_url}: {e}")

    # Load image from base64 data
    elif req.image_data:
        try:
            import base64
            image_bytes = base64.b64decode(req.image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_uri = "base64_data"
        except Exception as e:
            print(f"[WARN] Failed to decode base64 image data: {e}")

    return Sample(
        sample_id=sample_id,
        source="http",
        text=req.text,
        image=image,
        image_uri=image_uri,
        metadata=req.metadata,
        label=None,
    )


def route_response_from_result(result: InferenceResult) -> RouteResponse:
    """
    Convert InferenceResult to RouteResponse.

    Args:
        result: Inference result

    Returns:
        HTTP response object
    """
    return RouteResponse(
        sample_id=result.sample.sample_id,
        chosen_model=result.router_decision.chosen_model,
        probs=result.router_decision.probs,
        router_inference_ms=result.router_decision.inference_ms,
    )
