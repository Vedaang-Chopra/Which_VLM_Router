"""
Router model definition and checkpoint loading.

This module defines the MultimodalRouter architecture and utilities
for loading trained checkpoints.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModel
from typing import Dict, Any
from .config import RouterConfig


class MultimodalRouter(nn.Module):
    """
    Multimodal router that combines vision and text features to predict
    the best VLM for a given input.

    Architecture:
    - Vision encoder: CLIP ViT
    - Text encoder: DistilBERT or similar
    - Fusion: Concat + MLP
    - Classifier: Linear layer to num_models
    """

    def __init__(self, config: RouterConfig, num_models: int):
        """
        Initialize the multimodal router.

        Args:
            config: Router configuration
            num_models: Number of VLM models to route between
        """
        super().__init__()
        self.config = config
        self.num_models = num_models

        # Vision encoder (CLIP)
        vision_encoder_name = getattr(config, 'vision_encoder_name', 'openai/clip-vit-base-patch32')
        self.vision = CLIPVisionModel.from_pretrained(vision_encoder_name)
        vision_hidden_size = self.vision.config.hidden_size

        # Text encoder
        text_encoder_name = getattr(config, 'text_encoder_name', 'distilbert-base-uncased')
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        text_hidden_size = self.text_encoder.config.hidden_size

        # Projections to common dimension
        hidden_dim = max(vision_hidden_size, text_hidden_size)
        self.vision_proj = nn.Linear(vision_hidden_size, hidden_dim)
        self.text_proj = nn.Linear(text_hidden_size, hidden_dim)

        # Fusion MLP
        fused_dim = hidden_dim * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_models)

        # Freeze vision encoder if specified
        freeze_vision = getattr(config, 'freeze_vision', True)
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        # Freeze text encoder if specified
        freeze_text = getattr(config, 'freeze_text_encoder', False)
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the router.

        Args:
            pixel_values: Image tensor [B, C, H, W]
            input_ids: Text token IDs [B, T]
            attention_mask: Text attention mask [B, T]

        Returns:
            Logits of shape [B, num_models]
        """
        # Vision encoding
        vision_outputs = self.vision(pixel_values=pixel_values)
        vision_emb = vision_outputs.pooler_output
        vision_emb = self.vision_proj(vision_emb)

        # Text encoding
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get pooled text representation
        if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            text_emb = text_outputs.pooler_output
        else:
            # Mean pooling over sequence
            last_hidden = text_outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            text_emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        text_emb = self.text_proj(text_emb)

        # Fusion
        fused = torch.cat([vision_emb, text_emb], dim=-1)
        fused = self.fusion_mlp(fused)

        # Classification
        logits = self.classifier(fused)
        return logits


def load_router(config: RouterConfig, num_models: int) -> nn.Module:
    """
    Create router model, load checkpoint, and prepare for inference.

    Args:
        config: Router configuration
        num_models: Number of VLM models

    Returns:
        Router model ready for inference (on device, in eval mode)
    """
    # Set thread count for CPU inference
    if config.device == "cpu":
        torch.set_num_threads(config.num_threads)

    # Create model
    model = MultimodalRouter(config, num_models=num_models)

    # Load checkpoint
    state = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(state, dict):
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            # Assume it's just the state dict directly
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)

    # Set dtype
    if config.dtype == "float16":
        model = model.half()
    # float32 is default, no conversion needed

    # Move to device
    model.to(config.device)

    # Set to eval mode
    model.eval()

    return model
