"""
Public API for Artemis Router Module.

This module exposes the main functionality for routing requests to VLM models.
It supports initialization of different router architectures (Classical, Pairwise, Reward).
"""

import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

# Common config loader
from artemis_final.common.config_loader import GlobalConfig, get_base_dir, load_global_config

# Import Router Service
from .router_service import RouterService

# Import specific inference classes for advanced usage (e.g. notebooks)
# Import specific inference classes
# Using relative imports suitable for package structure
from .core.inference_reward_router import RewardRouterInference
from .core.legacy.inference_classical_router import ClassicalRouterInference
from .core.legacy.inference_pairwise_router import PairwiseRouterInference

logger = logging.getLogger(__name__)

# Singleton Entry Point
_GLOBAL_ROUTER_SERVICE: Optional[RouterService] = None

def init_router(config_path: Optional[str] = None):
    """
    Initialize the global Default Router Service (usually Reward Router).
    
    Args:
        config_path: Optional path to config yaml.
    """
    global _GLOBAL_ROUTER_SERVICE
    
    # Load Config
    if config_path:
        cfg = load_global_config(config_path)
    else:
        # Defaults to finding config.yaml in standard location
        cfg = load_global_config() 
        
    _GLOBAL_ROUTER_SERVICE = RouterService(cfg)
    logger.info("Global Router initialized.")

def route_request(prompt: str, mode: str = "balanced", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Route a request using the global initialized router.
    
    Args:
        prompt: User query/prompt.
        mode: Routing mode ('accuracy', 'cheap', 'fast', 'balanced').
        metadata: Extra info.
        
    Returns:
        Dict containing decision, probs, weights, etc.
    """
    if _GLOBAL_ROUTER_SERVICE is None:
        raise RuntimeError("Router not initialized. Call init_router() first.")
        
    return _GLOBAL_ROUTER_SERVICE.predict(prompt, mode, metadata)

def route_batch(prompts: List[str], modes: Optional[List[str]] = None, metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Route a batch of requests.
    
    Args:
        prompts: List of user prompts.
        modes: Optional list of modes (accuracy, cheap, etc).
        metadata_list: Optional list of metadata dicts.
        
    Returns:
        List of result dicts.
    """
    if _GLOBAL_ROUTER_SERVICE is None:
        raise RuntimeError("Router not initialized. Call init_router() first.")
        
    return _GLOBAL_ROUTER_SERVICE.route_batch(prompts, modes, metadata_list)

# -----------------------------------------------------------------------------
# Advanced API for Notebooks / Research
# -----------------------------------------------------------------------------

def load_router_from_checkpoint(
    router_type: str, 
    checkpoint_path: Optional[str] = None, 
    device: str = "cpu",
    verbose: bool = True
):
    """
    Load a specific router architecture from a checkpoint.
    
    Args:
        router_type: 'classical', 'pairwise', or 'reward'
        checkpoint_path: Path to the .pt file. If None, uses artemis.yaml.
        device: 'cpu', 'cuda', 'mps'
        verbose: Print loading info
        
    Returns:
        The inference object for that router type.
    """
    if checkpoint_path is None:
        if verbose:
            print(f"No checkpoint path provided. Loading from default config...")
        cfg = load_global_config()
        checkpoint_path = cfg.router.checkpoint_path
        
    checkpoint_path = str(checkpoint_path)
    
    if router_type.lower() == "classical":
        return ClassicalRouterInference(checkpoint_path, device, verbose)
    elif router_type.lower() == "pairwise":
        return PairwiseRouterInference(checkpoint_path, device, verbose)
    elif router_type.lower() == "reward":
        return RewardRouterInference(checkpoint_path, device, verbose)
    else:
        raise ValueError(f"Unknown router_type: {router_type}. Must be 'classical', 'pairwise', or 'reward'.")
