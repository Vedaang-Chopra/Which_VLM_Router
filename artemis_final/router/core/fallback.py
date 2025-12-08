"""
Confidence-based fallback logic for the Artemis Router.

When the router is uncertain (low confidence), this module provides
fallback strategies to improve routing decisions:
- Top-K selection
- Accuracy-weighted fallback
- Size-based fallback (prefer larger/more accurate models)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    # Minimum confidence threshold (below this triggers fallback)
    confidence_threshold: float = 0.3
    
    # How many top models to consider in fallback
    top_k: int = 2
    
    # Whether to prefer larger (more accurate) models in fallback
    prefer_larger_on_uncertainty: bool = True
    
    # Model size ranking (smallest to largest)
    model_size_ranking: List[str] = None
    
    def __post_init__(self):
        if self.model_size_ranking is None:
            self.model_size_ranking = [
                "qwen2_5_vl_3b",      # Smallest
                "deepseek_ocr",
                "qwen2_5_vl_7b",
                "qwen3_vl_8b_thinking",
                "gemma_3_27b",         # Largest
            ]


@dataclass
class FallbackResult:
    """Result of a fallback decision."""
    chosen_model: str
    original_model: str
    fallback_triggered: bool
    fallback_reason: Optional[str]
    confidence: float
    top_k_models: List[Tuple[str, float]]


class RouterFallback:
    """
    Handles fallback logic when router confidence is low.
    
    Usage:
        fallback = RouterFallback(config=FallbackConfig())
        result = fallback.apply(rewards, original_choice, stats_registry)
    """
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
    
    def compute_confidence(self, rewards: Dict[str, float]) -> float:
        """
        Compute confidence score from rewards.
        
        Confidence is based on:
        1. The gap between top-1 and top-2 models
        2. The absolute value of the top reward
        
        Returns:
            float: Confidence score in [0, 1]
        """
        if not rewards:
            return 0.0
        
        sorted_rewards = sorted(rewards.values(), reverse=True)
        
        if len(sorted_rewards) < 2:
            return 1.0  # Only one model, fully confident
        
        top_1 = sorted_rewards[0]
        top_2 = sorted_rewards[1]
        
        # Gap-based confidence: larger gap = more confident
        gap = top_1 - top_2
        
        # Normalize gap (assuming rewards are typically in [0, 1])
        # A gap of 0.2+ is considered high confidence
        gap_confidence = min(gap / 0.2, 1.0)
        
        # Absolute confidence: higher top reward = more confident
        abs_confidence = min(max(top_1, 0), 1.0)
        
        # Combined confidence (weighted average)
        confidence = 0.6 * gap_confidence + 0.4 * abs_confidence
        
        return max(0.0, min(1.0, confidence))
    
    def get_top_k_models(
        self, 
        rewards: Dict[str, float], 
        k: int = 2
    ) -> List[Tuple[str, float]]:
        """Get top-K models sorted by reward."""
        sorted_models = sorted(
            rewards.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_models[:k]
    
    def apply(
        self,
        rewards: Dict[str, float],
        original_choice: str,
        stats_registry: Optional[Any] = None,
        task_type: Optional[str] = None,
    ) -> FallbackResult:
        """
        Apply fallback logic if confidence is low.
        
        Args:
            rewards: Dict of model_name -> reward score
            original_choice: The router's original choice
            stats_registry: Optional stats for accuracy-weighted fallback
            task_type: Optional task type for task-specific fallback
            
        Returns:
            FallbackResult with final decision
        """
        confidence = self.compute_confidence(rewards)
        top_k = self.get_top_k_models(rewards, self.config.top_k)
        
        # No fallback needed if confidence is high enough
        if confidence >= self.config.confidence_threshold:
            return FallbackResult(
                chosen_model=original_choice,
                original_model=original_choice,
                fallback_triggered=False,
                fallback_reason=None,
                confidence=confidence,
                top_k_models=top_k,
            )
        
        logger.info(
            f"Low confidence ({confidence:.3f} < {self.config.confidence_threshold}), "
            f"triggering fallback. Original: {original_choice}"
        )
        
        # FALLBACK STRATEGY 1: Use stats_registry for accuracy-weighted selection
        if stats_registry is not None and task_type:
            chosen = self._accuracy_weighted_fallback(top_k, stats_registry, task_type)
            if chosen:
                return FallbackResult(
                    chosen_model=chosen,
                    original_model=original_choice,
                    fallback_triggered=True,
                    fallback_reason="accuracy_weighted",
                    confidence=confidence,
                    top_k_models=top_k,
                )
        
        # FALLBACK STRATEGY 2: Prefer larger models (more accurate)
        if self.config.prefer_larger_on_uncertainty:
            chosen = self._size_based_fallback(top_k)
            return FallbackResult(
                chosen_model=chosen,
                original_model=original_choice,
                fallback_triggered=True,
                fallback_reason="prefer_larger",
                confidence=confidence,
                top_k_models=top_k,
            )
        
        # FALLBACK STRATEGY 3: Just pick top-1 from top-K
        chosen = top_k[0][0] if top_k else original_choice
        return FallbackResult(
            chosen_model=chosen,
            original_model=original_choice,
            fallback_triggered=True,
            fallback_reason="top_k_selection",
            confidence=confidence,
            top_k_models=top_k,
        )
    
    def _accuracy_weighted_fallback(
        self,
        top_k: List[Tuple[str, float]],
        stats_registry: Any,
        task_type: str,
    ) -> Optional[str]:
        """
        Choose model based on historical accuracy for this task.
        
        Use stats_registry to get expected accuracy per model,
        pick the one with highest accuracy among top-K.
        """
        try:
            best_model = None
            best_accuracy = -1.0
            
            for model_name, _ in top_k:
                # Try to get accuracy from stats registry
                try:
                    accuracy = stats_registry.get_expected_accuracy(
                        task_type=task_type,
                        model_name=model_name
                    )
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_name
                except Exception:
                    continue
            
            return best_model
        except Exception as e:
            logger.warning(f"Accuracy-weighted fallback failed: {e}")
            return None
    
    def _size_based_fallback(
        self,
        top_k: List[Tuple[str, float]],
    ) -> str:
        """
        Among top-K, pick the largest model.
        
        Larger models are generally more accurate, so this
        is a safe choice when uncertain.
        """
        top_k_names = {model for model, _ in top_k}
        
        # Find largest model among top-K
        for model_name in reversed(self.config.model_size_ranking):
            if model_name in top_k_names:
                logger.info(f"Size-based fallback: choosing {model_name}")
                return model_name
        
        # Fallback to first in top-K
        return top_k[0][0] if top_k else self.config.model_size_ranking[-1]


def create_fallback_handler(
    confidence_threshold: float = 0.3,
    top_k: int = 2,
    prefer_larger: bool = True,
) -> RouterFallback:
    """
    Factory function to create a fallback handler.
    
    Args:
        confidence_threshold: Threshold below which fallback triggers
        top_k: Number of top models to consider
        prefer_larger: Whether to prefer larger models on uncertainty
        
    Returns:
        Configured RouterFallback instance
    """
    config = FallbackConfig(
        confidence_threshold=confidence_threshold,
        top_k=top_k,
        prefer_larger_on_uncertainty=prefer_larger,
    )
    return RouterFallback(config)
