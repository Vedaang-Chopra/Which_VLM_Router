"""
Core data structures for the Artemis Router.

These types unify DB-loaded, HTTP, and synthetic samples.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from PIL import Image


@dataclass
class Sample:
    """
    Unified sample representation for all input sources (DB, HTTP, synthetic).

    Attributes:
        sample_id: Unique identifier for this sample
        source: Origin of the sample ("db", "http", "synthetic")
        text: Text/question content
        image: PIL Image object (if available)
        image_uri: Path or URL to image (for logging)
        metadata: Additional metadata (split, dataset_name, etc.)
        label: Ground truth label/answer (if available)
    """
    sample_id: str
    source: str  # "db", "http", "synthetic"
    text: str
    image: Optional[Image.Image] = None
    image_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None


@dataclass
class RouterDecision:
    """
    Router's routing decision for a single sample.

    Attributes:
        sample_id: Sample identifier
        chosen_model: Top-1 model selected by router
        probs: Dictionary mapping model_name -> probability
        raw_logits: Raw logit values from router
        model_order: Order of models (matches logits/probs indices)
        inference_ms: Router inference latency in milliseconds
    """
    sample_id: str
    chosen_model: str
    probs: Dict[str, float]
    raw_logits: List[float]
    model_order: List[str]
    inference_ms: float


@dataclass
class InferenceResult:
    """
    Complete result of routing a sample.

    Contains both the input sample and the router's decision.
    In future phases, can add VLM model outputs here.

    Attributes:
        sample: Input sample
        router_decision: Router's decision
    """
    sample: Sample
    router_decision: RouterDecision


@dataclass
class LogRecord:
    """
    Record for SQL logging of router decisions.

    Attributes:
        timestamp: Unix timestamp
        sample_id: Sample identifier
        source: Sample source ("db", "http", "synthetic")
        text: Sample text
        image_uri: Image path/URL
        split: Dataset split (train/val/test)
        label: Ground truth label
        router_chosen_model: Model selected by router
        router_probs: Probability distribution over models
        router_inference_ms: Router latency
        extra_metadata: Additional metadata as JSON
    """
    timestamp: float
    sample_id: str
    source: str
    text: str
    image_uri: Optional[str]
    split: Optional[str]
    label: Optional[str]
    router_chosen_model: str
    router_probs: Dict[str, float]
    router_inference_ms: float
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
