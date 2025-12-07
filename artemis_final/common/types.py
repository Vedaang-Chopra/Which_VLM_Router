"""
Common type definitions shared across modules.
Uses Pydantic for validation and serialization.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

# ============================================================================
# Router Types
# ============================================================================

class RouterRequest(BaseModel):
    """Input to the router service."""
    request_id: str
    prompt: str
    mode: str = "balanced"
    metadata: Optional[Dict[str, Any]] = None

class RouterDecision(BaseModel):
    """Output from the router service."""
    request_id: str
    chosen_model: str
    model_probs: Dict[str, float] = Field(default_factory=dict)
    mode: str
    inference_ms: float = 0.0

# ============================================================================
# Load Balancer Types
# ============================================================================

class RoutedRequest(BaseModel):
    """Request after router decision, input to Load Balancer."""
    request_id: str
    router_decision: RouterDecision
    messages: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

class LBDecision(BaseModel):
    """Output from the Load Balancer."""
    request_id: str
    final_model: str
    router_preferred_model: str
    was_overridden: bool = False
    override_reason: Optional[str] = None
    queue_depth: int = 0
    estimated_latency_ms: float = 0.0

# ============================================================================
# Inference Types
# ============================================================================

class InferenceRequest(BaseModel):
    """Request to the Inference Service."""
    model_name: str
    messages: List[Dict[str, Any]]
    temperature: float = 0.7
    max_tokens: int = 512

class InferenceResponse(BaseModel):
    """Response from the Inference Service."""
    model_name: str
    content: str
    finish_reason: str = "stop"
    usage: Dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None

# ============================================================================
# Data Collection Types
# ============================================================================

class CollectedSample(BaseModel):
    """Represents a logged sample in the database."""
    id: Optional[int] = None
    request_id: str
    created_at: Optional[datetime] = None
    router_mode: str
    input_messages: List[Dict[str, Any]]
    chosen_model: str
    router_decision: Dict[str, Any]
    lb_decision: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None

class CollectedResponse(BaseModel):
    """Represents a logged response in the database."""
    id: Optional[int] = None
    sample_id: int
    created_at: Optional[datetime] = None
    model_name: str
    raw_response: Dict[str, Any]
    normalized_output: Optional[Dict[str, Any]] = None
    latency_ms: Optional[int] = None
    cost_cents: Optional[float] = None
    score: Optional[float] = None
    error: Optional[str] = None

class FeedbackRecord(BaseModel):
    """Represents feedback on a sample."""
    id: Optional[int] = None
    sample_id: int
    created_at: Optional[datetime] = None
    feedback_params: Dict[str, Any]
