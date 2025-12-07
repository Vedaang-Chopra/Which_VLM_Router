"""
Pipeline module: Wires together all services and provides high-level handlers.
This is the glue layer that orchestrates Router, LoadBalancer, Inference, and DataCollection.
"""
import logging
import time
import uuid
from typing import Dict, Any

from common.config_loader import load_global_config, GlobalConfig, get_base_dir
from common.types import RouterRequest, RouterDecision, LBDecision
from router.router_service import RouterService
from load_balancer.load_balancer_service import LoadBalancerService
from inference_engine.inference_service import InferenceService
from data_loop.collector import DataCollector
from router_train.service import Retrainer

from .schemas import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    ChatMessage, ChatCompletionUsage, RouterMetadata, FeedbackRequest
)

logger = logging.getLogger(__name__)

def init_system() -> Dict[str, Any]:
    """
    Initialize all system services.
    
    Returns:
        Dict with instantiated services:
            - cfg: GlobalConfig
            - router: RouterService
            - lb: LoadBalancerService
            - inference: InferenceService
            - collector: DataCollector
            - retrainer: Retrainer
    """
    logger.info("Initializing Artemis VLM Router system...")
    
    cfg = load_global_config()
    base_dir = get_base_dir()
    
    # Data Collector (needs DB access)
    collector = DataCollector(cfg)
    
    # Router Service
    router = RouterService(cfg)
    
    # Load Balancer Service
    lb = LoadBalancerService(cfg)
    
    # Inference Service
    inference = InferenceService(cfg, base_dir)
    
    # Retrainer
    retrainer = Retrainer(cfg, collector)
    
    logger.info("All services initialized successfully.")
    
    return {
        "cfg": cfg,
        "router": router,
        "lb": lb,
        "inference": inference,
        "collector": collector,
        "retrainer": retrainer,
    }

def handle_chat_completion(req: ChatCompletionRequest, services: Dict[str, Any]) -> ChatCompletionResponse:
    """
    Handle a /v1/chat/completions request through the full pipeline.
    
    Steps:
        1. Router: Predict best model
        2. Load Balancer: Apply SLA/load logic
        3. Inference: Call the chosen model
        4. Data Collection: Log everything
        5. Return formatted response
    """
    request_id = str(uuid.uuid4())
    router_svc: RouterService = services["router"]
    lb_svc: LoadBalancerService = services["lb"]
    inf_svc: InferenceService = services["inference"]
    collector: DataCollector = services["collector"]
    
    # Extract prompt from messages
    prompt = ""
    messages_raw = []
    for m in req.messages:
        messages_raw.append({"role": m.role, "content": m.content})
        if m.role == "user":
            if isinstance(m.content, str):
                prompt += m.content + " "
    prompt = prompt.strip()
    
    # Step 1: Router
    router_start = time.time()
    try:
        router_result = router_svc.predict(prompt, mode=req.router_mode or "balanced")
    except Exception as e:
        logger.error(f"Router failed: {e}")
        router_result = {"chosen_model": "qwen2_5_vl_7b", "rewards": {}, "mode": req.router_mode, "inference_ms": 0}
    router_ms = (time.time() - router_start) * 1000
    
    router_decision = RouterDecision(
        request_id=request_id,
        chosen_model=router_result.get("chosen_model", "unknown"),
        model_probs=router_result.get("rewards", {}),
        mode=router_result.get("mode", "balanced"),
        inference_ms=router_ms
    )
    
    # Step 2: Load Balancer
    lb_result = lb_svc.schedule(
        sample_id=request_id,
        task_type="vlm",
        router_probs=router_decision.model_probs,
        preferred_model=router_decision.chosen_model
    )
    
    final_model = lb_result.get("chosen_model", router_decision.chosen_model)
    was_overridden = final_model != router_decision.chosen_model
    
    lb_decision = LBDecision(
        request_id=request_id,
        final_model=final_model,
        router_preferred_model=router_decision.chosen_model,
        was_overridden=was_overridden,
        estimated_latency_ms=lb_result.get("total_latency_ms", 0)
    )
    
    # Step 3: Inference
    inf_start = time.time()
    try:
        inf_result = inf_svc.call_model(
            model_name=final_model,
            prompt=prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        content = inf_result.get("text") or inf_result.get("content") or ""
        finish_reason = inf_result.get("finish_reason", "stop")
        usage = inf_result.get("usage", {})
        error = None
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        content = f"Error: {str(e)}"
        finish_reason = "error"
        usage = {}
        error = str(e)
        inf_result = {"error": str(e)}
    inf_ms = (time.time() - inf_start) * 1000
    
    # Step 4: Log to DB
    try:
        sample_id = collector.log_sample_start(
            request_id=request_id,
            router_mode=req.router_mode or "balanced",
            input_messages=messages_raw,
            router_decision=router_decision.model_dump(),
            lb_decision=lb_decision.model_dump()
        )
        collector.log_model_response(
            sample_id=sample_id,
            model_name=final_model,
            raw_response=inf_result,
            normalized_output={"content": content},
            latency_ms=int(inf_ms),
            cost_cents=None,
            score=None,
            error=error
        )
        # Record outcome for LB stats
        lb_svc.record_outcome(lb_decision, {"latency_ms": inf_ms, "success": error is None})
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
    
    # Step 5: Build response
    return ChatCompletionResponse(
        id=request_id,
        object="chat.completion",
        created=int(time.time()),
        model=final_model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason=finish_reason
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0)
        ),
        router_metadata=RouterMetadata(
            router_preferred_model=router_decision.chosen_model,
            final_model=final_model,
            was_overridden=was_overridden,
            model_probs=router_decision.model_probs,
            router_inference_ms=router_ms
        )
    )

def handle_feedback(req: FeedbackRequest, services: Dict[str, Any]) -> int:
    """Handle feedback submission."""
    collector: DataCollector = services["collector"]
    
    # Resolve sample_id - it might be the request_id string or numeric ID
    sample_id = req.sample_id
    if isinstance(sample_id, str):
        # Look up by request_id
        sample_id = collector.get_sample_id_by_request_id(sample_id)
        if sample_id is None:
            raise ValueError(f"Sample not found for request_id: {req.sample_id}")
    
    return collector.log_feedback(sample_id, req.params)

def trigger_retrain(services: Dict[str, Any]) -> str:
    """
    Trigger retraining and hot-swap the router.
    
    Returns:
        Path to new checkpoint or error message.
    """
    retrainer: Retrainer = services["retrainer"]
    router_svc: RouterService = services["router"]
    
    try:
        new_checkpoint = retrainer.retrain_once()
        if new_checkpoint:
            router_svc.reload_model(new_checkpoint)
            return new_checkpoint
        return "No new checkpoint (insufficient data or other reason)"
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise e
