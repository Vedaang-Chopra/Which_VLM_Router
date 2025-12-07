"""
InferenceService: Wrapper around the WhichVLM client for model inference.
Uses common.config_loader for configuration.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Import existing inference logic
try:
    from .client import WhichVLMClient
except ImportError:
    from inference_engine.client import WhichVLMClient

from common.config_loader import GlobalConfig

logger = logging.getLogger(__name__)

class InferenceService:
    """
    Service wrapper for VLM inference.
    Provides a clean interface for calling any configured model.
    """
    
    def __init__(self, cfg: GlobalConfig, base_dir: Optional[Path] = None):
        """
        Initialize the Inference Service.
        
        Args:
            cfg: GlobalConfig from common.config_loader
            base_dir: Base directory for resolving paths
        """
        self.cfg = cfg
        self.base_dir = Path(base_dir) if base_dir else Path(cfg._base_dir)
        self.models_file = self.base_dir / cfg.inference.models_file
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> Optional[WhichVLMClient]:
        """Initialize the WhichVLM client."""
        logger.info(f"Initializing InferenceService with models: {self.models_file}")
        
        if not self.models_file.exists():
            logger.warning(f"Models config not found: {self.models_file}. Inference will fail.")
            return None
        
        try:
            return WhichVLMClient.from_yaml(str(self.models_file))
        except Exception as e:
            logger.error(f"Failed to initialize inference client: {e}")
            return None

    def call_model(self, 
                   model_name: str, 
                   prompt: str, 
                   image_path: Optional[str] = None,
                   system_prompt: Optional[str] = None,
                   temperature: float = 0.7,
                   max_tokens: int = 512) -> Dict[str, Any]:
        """
        Call a specific model for inference.
        
        Args:
            model_name: Name of the model to call
            prompt: User prompt
            image_path: Optional path to image (for VLM)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Dict with keys: text/content, finish_reason, usage
        """
        if self.client is None:
            raise RuntimeError("Inference client not initialized")
        
        try:
            if image_path:
                # Use VLM suite
                results = self.client.vlm.run_image(
                    image_path=image_path,
                    prompt=prompt,
                    models=[model_name],
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Use LLM suite
                results = self.client.llm.run_single(
                    prompt=prompt,
                    models=[model_name],
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            if model_name not in results:
                raise RuntimeError(f"Model {model_name} did not return a response")
            
            return results[model_name]
            
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {e}")
            raise e

    def list_models(self) -> list:
        """List available models."""
        if self.client is None:
            return []
        return self.client.list_models()
