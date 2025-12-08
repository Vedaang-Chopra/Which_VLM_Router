"""
RouterService: Wrapper around the reward router inference logic.
Uses common.config_loader for configuration.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Import existing router logic
try:
    from .artemis_router.inference_reward_router import RewardRouterInference
except ImportError:
    from artemis_router.inference_reward_router import RewardRouterInference

from common.config_loader import GlobalConfig, get_base_dir

logger = logging.getLogger(__name__)

class RouterService:
    """
    Service wrapper for the Artemis reward router.
    Provides a clean interface for routing requests to the best model.
    """
    
    def __init__(self, cfg: GlobalConfig):
        """
        Initialize the Router Service.
        
        Args:
            cfg: GlobalConfig from common.config_loader
        """
        self.cfg = cfg
        self.base_dir = Path(cfg._base_dir)
        self.checkpoint_path = self.base_dir / cfg.router.checkpoint_path
        self.device = cfg.router.device
        self.engine = None
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Load the router model."""
        logger.info(f"Initializing RouterService with checkpoint: {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {self.checkpoint_path}. Router will use fallback.")
            return
        
        try:
            self.engine = RewardRouterInference(
                checkpoint_path=str(self.checkpoint_path),
                device=self.device,
                verbose=True
            )
            logger.info("Router engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize router: {e}")
            # Continue without engine - will use fallback

    def predict(self, prompt: str, mode: str = "balanced", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route a prompt to the best model.
        
        Args:
            prompt: User prompt text
            mode: Routing mode (accuracy, cheap, fast, balanced)
            metadata: Optional additional metadata
            
        Returns:
            Dict[str, Any]: A dictionary containing routing details:
            - `chosen_model` (str): The selected model name.
            - `rewards` (Dict[str, float]): Scores/logits for each available model.
            - `mode` (str): The routing mode applied.
            - `inference_ms` (float): Latency of the routing step.
        """
        if self.engine is None:
            # Fallback when no model is loaded
            return {
                "chosen_model": "qwen2_5_vl_7b",  # Default fallback
                "rewards": {},
                "mode": mode,
                "inference_ms": 0.0
            }
        
        return self.engine.route(prompt, mode=mode, metadata=metadata)

    def reload_model(self, new_checkpoint_path: Optional[str] = None):
        """
        Hot-reload the router model.
        
        Args:
            new_checkpoint_path: Path to new checkpoint (uses config default if None)
        """
        if new_checkpoint_path:
            self.checkpoint_path = Path(new_checkpoint_path)
            if not self.checkpoint_path.is_absolute():
                self.checkpoint_path = self.base_dir / self.checkpoint_path
        
        logger.info(f"Reloading router from: {self.checkpoint_path}")
        self._initialize_engine()
