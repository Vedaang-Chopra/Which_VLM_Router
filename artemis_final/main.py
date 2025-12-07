import logging
import uuid
import os
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Internal Imports
from common.config_loader import SystemConfig, get_default_paths
from common.db import get_db_engine
from router.router_service import RouterService
from load_balancer.load_balancer_service import LoadBalancerService
from inference_engine.inference_service import InferenceService
from data_loop.collector import DataCollector
from data_loop.retrainer import Retrainer

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("artemis_main")

# Models for API
class ChatMessage(BaseModel):
    role: str
    content: str | List[Dict[str, Any]] # handle multimodal content

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "router-auto"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    # Custom fields for router
    router_mode: Optional[str] = "balanced"
    
class FeedbackRequest(BaseModel):
    sample_id: str
    params: Dict[str, Any] # score, ground_truth, etc.

# App Factory
def create_app() -> FastAPI:
    app = FastAPI(title="Artemis VLM Router System", version="1.0.0")
    
    # Load Global State
    try:
        paths = get_default_paths()
        # Allow override via env vars
        if os.getenv("ROUTER_CONFIG_PATH"): paths['router'] = os.getenv("ROUTER_CONFIG_PATH")
        if os.getenv("LB_CONFIG_PATH"): paths['lb'] = os.getenv("LB_CONFIG_PATH")
        if os.getenv("MODELS_CONFIG_PATH"): paths['inference'] = os.getenv("MODELS_CONFIG_PATH")
        
        config = SystemConfig.load(
            paths['router'], 
            paths['lb'], 
            paths['inference']
        )
        
        # Initialize Services
        app.state.config = config
        app.state.router = RouterService(config.router_config)
        app.state.lb = LoadBalancerService(config.lb_config)
        app.state.inference = InferenceService(config.inference_config_path)
        app.state.collector = DataCollector(config.router_config.db_url)
        app.state.retrainer = Retrainer(config)
        
        logger.info("All services initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # We don't raise here to allow app to start and report health error? No, let's crash fast.
        raise e

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, background_tasks: BackgroundTasks):
        sample_id = str(uuid.uuid4())
        
        # Extract Prompt
        # Basic logic: concat user messages. Real implementation needs robust parsing.
        prompt = ""
        image_path = None # Logic to extract image not implemented yet
        
        for m in req.messages:
            if m.role == "user":
                if isinstance(m.content, str):
                    prompt += m.content + "\n"
                # TODO: handle list of content for images
        
        prompt = prompt.strip()
        if not prompt:
            raise HTTPException(400, "Empty prompt")
            
        # 1. Router
        try:
            router_result = app.state.router.predict(prompt, mode=req.router_mode)
        except Exception as e:
            logger.error(f"Router failed: {e}")
            raise HTTPException(500, "Router internal error")

        best_model = router_result['chosen_model']
        router_probs = router_result.get('rewards', {})
        
        # 2. Load Balancer
        lb_decision = app.state.lb.schedule(
            sample_id=sample_id,
            task_type="vlm", 
            router_probs=router_probs,
            preferred_model=best_model
        )
        target_model = lb_decision['chosen_model']
        
        # 3. Inference
        try:
            inference_result = app.state.inference.call_model(
                model_name=target_model,
                prompt=prompt,
                image_path=image_path,
                temperature=req.temperature,
                max_tokens=req.max_tokens
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Log failure
            background_tasks.add_task(
                app.state.collector.log_request_async,
                sample_id, prompt, image_path, router_result, lb_decision, {"error": str(e)}
            )
            raise HTTPException(502, f"Model execution failed: {e}")
            
        # 4. Background Logging
        background_tasks.add_task(
            app.state.collector.log_request_async,
            sample_id, prompt, image_path, router_result, lb_decision, inference_result
        )
        
        # 5. Format Response
        # OpenAI format mimic
        resp_text = inference_result.get('text') or inference_result.get('content') or ""
        return {
            "id": sample_id,
            "object": "chat.completion",
            "created": 12345678,
            "model": target_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp_text
                },
                "finish_reason": inference_result.get('finish_reason', 'stop')
            }],
            "usage": inference_result.get('usage', {})
        }
        
    @app.post("/feedback")
    def submit_feedback(req: FeedbackRequest):
        """Submit feedback for a previous request."""
        score = req.params.get('score')
        text = req.params.get('text')
        
        if score is not None:
            app.state.collector.update_feedback(req.sample_id, score=float(score), text=text)
        return {"status": "received"}

    @app.post("/admin/retrain")
    async def trigger_retrain(background_tasks: BackgroundTasks):
        """Trigger a retraining job in background."""
        
        def _run_retrain_job():
            logger.info("Starting background retrain job...")
            try:
                new_checkpoint = app.state.retrainer.run_retraining(epochs=1)
                if new_checkpoint:
                    logger.info(f"Retraining success. New checkpoint: {new_checkpoint}")
                    # Hot-reload
                    app.state.router.reload_model(new_checkpoint)
                else:
                    logger.info("Retraining skipped (no data or other reason).")
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
                
        background_tasks.add_task(_run_retrain_job)
        return {"status": "retraining_started"}

    return app

# Entry point for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
