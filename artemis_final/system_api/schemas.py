"""
Pydantic schemas for the Artemis VLM Router FastAPI endpoints.
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

# ============================================================================
# Chat Completion API (OpenAI-compatible)
# ============================================================================

class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str
    content: Union[str, List[Dict[str, Any]]]  # str or multimodal content

class ChatCompletionRequest(BaseModel):
    """Request body for /v1/chat/completions."""
    messages: List[ChatMessage]
    model: Optional[str] = "router-auto"  # Ignored, router decides
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    router_mode: Optional[str] = "balanced"  # Custom: accuracy, cheap, fast, balanced

class ChatCompletionChoice(BaseModel):
    """A single choice in the completion response."""
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    """Token usage info."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class RouterMetadata(BaseModel):
    """Metadata about router decision."""
    router_preferred_model: str
    final_model: str
    was_overridden: bool = False
    model_probs: Dict[str, float] = Field(default_factory=dict)
    router_inference_ms: float = 0.0

class ChatCompletionResponse(BaseModel):
    """Response body for /v1/chat/completions."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
    router_metadata: Optional[RouterMetadata] = None

# ============================================================================
# Feedback API
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request body for /feedback."""
    sample_id: Union[int, str]  # ID from the original response
    params: Dict[str, Any]  # Must include "score": float, etc.

class FeedbackResponse(BaseModel):
    """Response body for /feedback."""
    status: str
    feedback_id: Optional[int] = None

# ============================================================================
# Admin API
# ============================================================================

class RetrainResponse(BaseModel):
    """Response body for /admin/retrain."""
    status: str
    message: str
    new_checkpoint: Optional[str] = None
