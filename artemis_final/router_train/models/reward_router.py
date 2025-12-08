"""
Reward router model: predicts reward for (query, model, mode) triples.

Architecture:
    - Text encoder (e.g., DistilBERT)
    - Model embedding layer
    - Mode embedding layer
    - MLP head for reward prediction
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

import sys
from pathlib import Path

# Add artemis_final's parent to path for package imports
_artemis_parent = Path(__file__).parent.parent.parent.parent
if str(_artemis_parent) not in sys.path:
    sys.path.insert(0, str(_artemis_parent))

from artemis_final.router_train.config import RouterModelConfig

logger = logging.getLogger(__name__)


class RewardRouterModel(nn.Module):
    """
    Multi-objective VLM router that predicts reward scores and task type.
    
    The model takes as input:
        - Text (tokenized query with metadata)
        - Model ID (which VLM model)
        - Mode ID (which reward mode: accuracy, cheap, fast, balanced)
        
    And outputs:
        - utility_hat: Scalar utility prediction
        - task_logits: Classification logits for task type
    """

    def __init__(
        self,
        config: RouterModelConfig,
        num_models: int,
        num_modes: int,
        num_tasks: int,
        text_encoder_hidden_size: Optional[int] = None,
    ):
        """
        Initialize reward router model.

        Args:
            config: Model configuration
            num_models: Number of VLM models to choose from
            num_modes: Number of reward modes
            num_tasks: Number of router tasks (for classification head)
            text_encoder_hidden_size: Hidden size of text encoder (auto-detected if None)
        """
        super().__init__()

        self.config = config
        self.num_models = num_models
        self.num_modes = num_modes
        self.num_tasks = num_tasks

        # Text encoder
        logger.info(f"Loading text encoder: {config.text_encoder_name}")
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_name)

        # Get text encoder hidden size
        if text_encoder_hidden_size is None:
            # Auto-detect from config
            if hasattr(self.text_encoder.config, "hidden_size"):
                text_encoder_hidden_size = self.text_encoder.config.hidden_size
            elif hasattr(self.text_encoder.config, "dim"):
                text_encoder_hidden_size = self.text_encoder.config.dim
            else:
                raise ValueError("Could not auto-detect text encoder hidden size. Please specify explicitly.")

        self.text_hidden_size = text_encoder_hidden_size
        
        # Freeze text encoder if requested
        if config.freeze_text_encoder:
            logger.info("Freezing text encoder parameters")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Embeddings
        self.model_embedding = nn.Embedding(num_models, config.model_emb_dim)
        self.mode_embedding = nn.Embedding(num_modes, config.mode_emb_dim)

        # Routing Head (Utility)
        # Input: [text, model, mode]
        input_dim = self.text_hidden_size + config.model_emb_dim + config.mode_emb_dim
        
        self.routing_mlp = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1) # scalar utility
        )
        
        # Task Head (Classification)
        # Input: [text] only
        self.task_head = nn.Sequential(
            nn.Linear(self.text_hidden_size, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_tasks)
        )

        logger.info(f"Model initialized with {self.count_parameters():,} trainable parameters")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model_id: torch.Tensor,
        mode_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            model_id: Model IDs [batch_size]
            mode_id: Mode IDs [batch_size]

        Returns:
            Dictionary with:
            - utility_hat: Predicted utility scores [batch_size]
            - task_logits: Task classification logits [batch_size, num_tasks]
        """
        # Encode text
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token (index 0) - aligning with notebook implementation
        if hasattr(outputs, "last_hidden_state"):
            h_text = outputs.last_hidden_state[:, 0, :]
        else:
            h_text = outputs[0][:, 0, :]
        
        # Task Prediction (Text only)
        task_logits = self.task_head(h_text)
        
        # Utility Prediction
        h_model = self.model_embedding(model_id)
        h_mode = self.mode_embedding(mode_id)
        
        h_combined = torch.cat([h_text, h_model, h_mode], dim=-1)
        utility_hat = self.routing_mlp(h_combined).squeeze(-1)
        
        return {
            "utility_hat": utility_hat,
            "task_logits": task_logits
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "config": self.config,
            "num_models": self.num_models,
            "num_modes": self.num_modes,
            "num_tasks": self.num_tasks,
            "text_hidden_size": self.text_hidden_size,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "RewardRouterModel":
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model to (default: auto-detect)

        Returns:
            Loaded model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = cls(
            config=checkpoint["config"],
            num_models=checkpoint["num_models"],
            num_modes=checkpoint["num_modes"],
            num_tasks=checkpoint.get("num_tasks", 30), # Default fall back if missing
            text_encoder_hidden_size=checkpoint.get("text_hidden_size"),
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        logger.info(f"Model loaded from: {path}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Parameters: {model.count_parameters():,}")

        return model

def create_model(
    config: RouterModelConfig,
    num_models: int,
    num_modes: int,
    num_tasks: int,
    device: Optional[str] = None,
) -> RewardRouterModel:
    """
    Convenience function to create and initialize a reward router model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RewardRouterModel(config, num_models, num_modes, num_tasks)
    model.to(device)
    return model
