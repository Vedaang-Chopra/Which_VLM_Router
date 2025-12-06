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

from config import RouterModelConfig

logger = logging.getLogger(__name__)


class RewardRouterModel(nn.Module):
    """
    Multi-objective VLM router that predicts reward scores.

    The model takes as input:
        - Text (tokenized query with metadata)
        - Model ID (which VLM model)
        - Mode ID (which reward mode: accuracy, cheap, fast, balanced)

    And outputs a scalar reward prediction.
    """

    def __init__(
        self,
        config: RouterModelConfig,
        num_models: int,
        num_modes: int,
        text_encoder_hidden_size: Optional[int] = None,
    ):
        """
        Initialize reward router model.

        Args:
            config: Model configuration
            num_models: Number of VLM models to choose from
            num_modes: Number of reward modes
            text_encoder_hidden_size: Hidden size of text encoder (auto-detected if None)
        """
        super().__init__()

        self.config = config
        self.num_models = num_models
        self.num_modes = num_modes

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
        logger.info(f"Text encoder hidden size: {self.text_hidden_size}")

        # Freeze text encoder if requested
        if config.freeze_text_encoder:
            logger.info("Freezing text encoder parameters")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Embedding layers
        self.model_embedding = nn.Embedding(num_models, config.model_emb_dim)
        self.mode_embedding = nn.Embedding(num_modes, config.mode_emb_dim)

        # MLP head
        self.mlp = self._build_mlp(
            input_dim=self.text_hidden_size + config.model_emb_dim + config.mode_emb_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
            activation=config.activation,
        )

        # Output layer
        self.output = nn.Linear(config.hidden_dim, 1)

        # Initialize weights
        self._init_weights()

        logger.info(f"Model initialized with {self.count_parameters():,} trainable parameters")

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ) -> nn.Module:
        """
        Build MLP head.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function name

        Returns:
            Sequential MLP module
        """
        # Get activation function
        if activation.lower() == "gelu":
            act_fn = nn.GELU
        elif activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []

        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            act_fn(),
            nn.Dropout(dropout),
        ])

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights for embedding and linear layers."""
        # Initialize embeddings
        nn.init.normal_(self.model_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mode_embedding.weight, mean=0.0, std=0.02)

        # Initialize MLP
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize output layer
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model_id: torch.Tensor,
        mode_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            model_id: Model IDs [batch_size]
            mode_id: Mode IDs [batch_size]

        Returns:
            Predicted rewards [batch_size]
        """
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Pool text representation
        if self.config.text_pooling == "cls":
            # Use [CLS] token (first token)
            h_text = text_outputs.last_hidden_state[:, 0, :]
        elif self.config.text_pooling == "mean":
            # Mean pooling (excluding padding)
            token_embeddings = text_outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            h_text = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling method: {self.config.text_pooling}")

        # Get embeddings
        h_model = self.model_embedding(model_id)  # [batch_size, model_emb_dim]
        h_mode = self.mode_embedding(mode_id)     # [batch_size, mode_emb_dim]

        # Concatenate
        h = torch.cat([h_text, h_model, h_mode], dim=-1)  # [batch_size, total_dim]

        # MLP
        h = self.mlp(h)  # [batch_size, hidden_dim]

        # Output
        reward = self.output(h).squeeze(-1)  # [batch_size]

        return reward

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_device(self) -> torch.device:
        """Get device of model."""
        return next(self.parameters()).device

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

        checkpoint = torch.load(path, map_location=device)

        model = cls(
            config=checkpoint["config"],
            num_models=checkpoint["num_models"],
            num_modes=checkpoint["num_modes"],
            text_encoder_hidden_size=checkpoint["text_hidden_size"],
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
    device: Optional[str] = None,
) -> RewardRouterModel:
    """
    Convenience function to create and initialize a reward router model.

    Args:
        config: Model configuration
        num_models: Number of VLM models
        num_modes: Number of reward modes
        device: Device to place model on (default: auto-detect)

    Returns:
        Initialized model
    """
    if device is None:
        if config:
            device = "auto"
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

    model = RewardRouterModel(config, num_models, num_modes)
    model.to(device)

    logger.info(f"Created model on device: {device}")

    return model
