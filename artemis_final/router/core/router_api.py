"""
Artemis Router API

This module provides a clean, high-level interface for using the Artemis VLM Router.
It is designed to be the single entry point for most router interactions.

Usage:
    from artemis_router.router_api import route_text, RouterService
    
    # Simple functional usage
    decision = route_text("What is in this image?", image=img, mode="balanced")
    
    # Service usage (dependency injection style)
    router = RouterService(config)
    decision = router.predict("prompt", mode="fast")
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from PIL import Image

# Re-export key classes for convenience
from .inference_reward_router import RewardRouterInference
from .schemas import Sample, RouterDecision, InferenceResult, LogRecord
from .traffic_simulator import run_traffic, TrafficConfig

# Import service logic (we might merge router_service logic here or keep it as a wrapper)
from ..router_service import RouterService

logger = logging.getLogger(__name__)

# --- Singleton for easy functional usage ---

class _RouterSingleton:
    """
    Lazy-loaded singleton to support simple functional API.
    Does not load model until first call.
    """
    _instance: Optional[RewardRouterInference] = None
    _default_checkpoint: str = "best_reward_router.pt" # Relative to checkpoints dir usually

    @classmethod
    def get_instance(cls) -> RewardRouterInference:
        if cls._instance is None:
            # Try to find safe default paths
            # Assuming we are in artemis_final/router/artemis_router/router_api.py
            # Checkpoints are usually in artemis_final/checkpoints/
            
            # Helper to find checkpoint
            current_dir = Path(__file__).parent  # artemis_router
            router_dir = current_dir.parent      # router
            artemis_final_dir = router_dir.parent
            
            checkpoint_path = artemis_final_dir / "checkpoints" / cls._default_checkpoint
            
            if not checkpoint_path.exists():
                logger.warning(f"Default checkpoint not found at {checkpoint_path}. "
                               f"Trying to find it in ../checkpoints relative to CWD.")
                # Fallback purely relative
                checkpoint_path = Path("../checkpoints") / cls._default_checkpoint

            logger.info(f"Initializing global router instance from {checkpoint_path}...")
            
            # Determine best device
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
            try:
                cls._instance = RewardRouterInference(
                    checkpoint_path=str(checkpoint_path),
                    device=device,
                    verbose=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize global router: {e}")
                raise RuntimeError(f"Could not load router checkpoint from {checkpoint_path}. "
                                   "Please use RouterService or RewardRouterInference directly with correct path.") from e
                                   
        return cls._instance

_GLOBAL_ROUTER = _RouterSingleton


# --- High-Level API Functions ---

def route_text(
    prompt: str,
    image: Optional[Image.Image] = None,
    mode: str = "balanced",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Route a text prompt (and optional image) to the best VLM.
    
    This is a convenience wrapper that manages a global router instance.
    
    Args:
        prompt (str): The user query or prompt.
        image (Optional[Image.Image]): Optional PIL Image. If provided, router considers image resolution.
        mode (str): Routing mode ('accuracy', 'cheap', 'fast', 'balanced'). Defaults to "balanced".
        metadata (Optional[Dict[str, Any]]): Optional metadata dict.
            Common keys include:
            - `task_type` (str): e.g. 'vqa', 'ocr'.
            - `source` (str): 'user', 'synthetic', etc.
            - `user_id` (str): For tracking user-specific behavior.
    
    Returns:
        Dict[str, Any]: Routing result with keys:
            - `chosen_model` (str): Name of the selected model.
            - `rewards` (Dict[str, float]): Predicted reward/score for each model.
            - `mode` (str): The routing mode used.
            - `inference_ms` (float): Time taken for routing inference in milliseconds.
    """
    router = _GLOBAL_ROUTER.get_instance()
    return router.route(prompt, image=image, mode=mode, metadata=metadata)


def route_sample(sample: Sample) -> InferenceResult:
    """
    Route a Sample object.
    
    Args:
        sample (Sample): A standardized Sample object (from DB, HTTP, etc.) containing
                         text, image, and metadata.
    
    Returns:
        InferenceResult: An object containing:
            - `sample`: The original input Sample.
            - `router_decision`: A RouterDecision object detailing the choice,
                                 probabilities, and model ranking.
    """
    router = _GLOBAL_ROUTER.get_instance()
    
    # Map 'Sample' to router arguments
    # Note: Sample.image is PIL Image, which router accepts directly
    result_dict = router.route(
        prompt=sample.text,
        image=sample.image,
        # Default to balanced if not in metadata, or user can set policy elsewhere
        # Ideally Sample would have a requested mode, but usually it's passed separately.
        # We will check metadata for mode, else default.
        mode=sample.metadata.get("mode", "balanced"),
        metadata=sample.metadata
    )
    
    # Convert dict to RouterDecision
    decision = RouterDecision(
        sample_id=sample.sample_id,
        chosen_model=result_dict['chosen_model'],
        probs=result_dict['rewards'], # In reward router, rewards ~ logits/probs conceptually
        raw_logits=list(result_dict['rewards'].values()),
        model_order=list(result_dict['rewards'].keys()),
        inference_ms=result_dict['inference_ms']
    )
    
    return InferenceResult(sample=sample, router_decision=decision)


def run_traffic_simulation(
    rps: float = 5.0,
    duration_sec: int = 30,
    source: str = "synthetic",
    verbose: bool = True
):
    """
    Run a simple traffic simulation to stress-test the router.
    
    Args:
        rps: Requests per second to simulate.
        duration_sec: How long to run.
        source: 'synthetic' generates random text/images.
        verbose: Print progress.
        
    Returns:
        Tuple (results, stats)
    """
    router = _GLOBAL_ROUTER.get_instance()
    
    # We need a route function that accepts a Sample and returns InferenceResult
    # route_sample fits this signature exactly!
    
    if source == "synthetic":
        traffic_cfg = TrafficConfig(
            synthetic_text_length=10,
            synthetic_image_shape=(224, 224, 3) 
        )
    else:
        traffic_cfg = None
        
    print(f"Starting simulation: {rps} RPS for {duration_sec}s...")
    results, stats = run_traffic(
        route_fn=route_sample,
        source=source,
        traffic_cfg=traffic_cfg,
        rps=rps,
        duration_sec=duration_sec,
        verbose=verbose
    )
    return results, stats

# Expose key components
__all__ = [
    "route_text",
    "route_sample",
    "run_traffic_simulation",
    "RewardRouterInference",
    "RouterService",
    "Sample",
    "RouterDecision",
    "InferenceResult",
]
