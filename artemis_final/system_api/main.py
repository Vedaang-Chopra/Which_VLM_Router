"""
FastAPI application for the Artemis VLM Router system.
Exposes: /health, /v1/chat/completions, /feedback, /admin/retrain
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks

from .schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    FeedbackRequest, FeedbackResponse,
    RetrainResponse
)
from .pipeline import init_system, handle_chat_completion, handle_feedback, trigger_retrain

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("artemis_api")

# Global services dict (initialized on startup)
SERVICES = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    global SERVICES
    logger.info("Starting Artemis VLM Router API...")
    try:
        SERVICES = init_system()
        logger.info("Services initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise e
    yield
    # Cleanup on shutdown (if needed)
    logger.info("Shutting down Artemis VLM Router API...")

app = FastAPI(
    title="Artemis VLM Router",
    description="Unified VLM routing, load balancing, and inference API with continuous learning.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    Routes the request through Router -> LoadBalancer -> Inference.
    Logs everything to the database.
    """
    if not SERVICES:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        return handle_chat_completion(req, SERVICES)
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """
    Submit feedback for a previous request.
    Requires sample_id (from response) and params (must include "score").
    """
    if not SERVICES:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        feedback_id = handle_feedback(req, SERVICES)
        return FeedbackResponse(status="received", feedback_id=feedback_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/retrain", response_model=RetrainResponse)
def admin_retrain(background_tasks: BackgroundTasks):
    """
    Trigger retraining of the router model.
    Runs in background, returns immediately.
    """
    if not SERVICES:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    def _do_retrain():
        try:
            result = trigger_retrain(SERVICES)
            logger.info(f"Retraining complete: {result}")
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    
    background_tasks.add_task(_do_retrain)
    return RetrainResponse(
        status="started",
        message="Retraining job started in background"
    )

# For running directly with: python -m system_api.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
