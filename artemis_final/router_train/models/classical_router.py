"""
Classical multi-class classification router for VLM routing.

Predicts best model using Cross-Entropy loss (hard labels) and KL divergence (soft labels).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ClassicalRouterModel(nn.Module):
    """
    Classical classification-based router.

    Architecture:
        1. Text encoder (DistilBERT) for sample text
        2. Learnable mode embeddings
        3. MLP to produce logits over models
        4. Training: CE loss on hard labels + KL loss on soft labels (teacher distribution)
    """

    def __init__(
        self,
        num_models: int,
        num_modes: int = 4,
        text_encoder_name: str = "distilbert-base-uncased",
        mode_embed_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_models: Number of VLM models (output classes)
            num_modes: Number of routing modes
            text_encoder_name: Hugging Face model name for text encoding
            mode_embed_dim: Dimension of mode embeddings
            hidden_dim: Hidden layer size in MLP
            dropout: Dropout probability
        """
        super().__init__()

        self.num_models = num_models
        self.num_modes = num_modes

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size

        # Mode embeddings
        self.mode_embeddings = nn.Embedding(num_modes, mode_embed_dim)

        # Classifier MLP
        combined_dim = text_dim + mode_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_models),  # Logits over models
        )

        logger.info(f"Initialized ClassicalRouterModel:")
        logger.info(f"  Models (classes): {num_models}, Modes: {num_modes}")
        logger.info(f"  Text encoder: {text_encoder_name} (dim={text_dim})")
        logger.info(f"  Mode embed: {mode_embed_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}, Dropout: {dropout}")

    def forward(
        self,
        sample_texts: List[str],
        mode_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to compute logits over models.

        Args:
            sample_texts: List of sample text strings (length B)
            mode_ids: Mode IDs, shape (B,)

        Returns:
            logits: Logits over models, shape (B, num_models)
        """
        device = next(self.parameters()).device

        # Encode text
        encoded = self.tokenizer(
            sample_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        text_outputs = self.text_encoder(**encoded)
        text_emb = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token (B, text_dim)

        # Embed mode
        mode_emb = self.mode_embeddings(mode_ids.to(device))  # (B, mode_embed_dim)

        # Combine and classify
        combined = torch.cat([text_emb, mode_emb], dim=1)  # (B, combined_dim)
        logits = self.classifier(combined)  # (B, num_models)

        return logits

    def compute_loss(
        self,
        sample_texts: List[str],
        mode_ids: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_labels: Optional[torch.Tensor] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined CE + KL loss.

        Loss = alpha * CE(logits, hard_labels) + (1-alpha) * KL(soft_labels || logits)

        Args:
            sample_texts: List of sample text strings (length B)
            mode_ids: Mode IDs, shape (B,)
            hard_labels: Hard label model IDs, shape (B,)
            soft_labels: Soft label distributions, shape (B, num_models), optional
            temperature: Temperature for KL divergence
            alpha: Weight for CE loss (1-alpha for KL loss)

        Returns:
            loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        logits = self.forward(sample_texts, mode_ids)  # (B, num_models)

        # Cross-entropy loss on hard labels
        ce_loss = F.cross_entropy(logits, hard_labels)

        loss_dict = {"ce_loss": ce_loss.item()}

        # KL divergence on soft labels (if provided)
        if soft_labels is not None:
            # Apply temperature scaling
            log_probs = F.log_softmax(logits / temperature, dim=1)
            soft_probs = soft_labels  # Already probabilities

            # KL divergence: sum(soft_probs * log(soft_probs / pred_probs))
            kl_loss = F.kl_div(log_probs, soft_probs, reduction="batchmean")

            loss_dict["kl_loss"] = kl_loss.item()

            # Combine losses
            total_loss = alpha * ce_loss + (1 - alpha) * kl_loss
            loss_dict["total_loss"] = total_loss.item()

            return total_loss, loss_dict
        else:
            # Only CE loss
            loss_dict["total_loss"] = ce_loss.item()
            return ce_loss, loss_dict

    def predict(
        self,
        sample_texts: List[str],
        mode_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict best model for each sample.

        Args:
            sample_texts: List of sample text strings (length B)
            mode_ids: Mode IDs, shape (B,)

        Returns:
            predictions: Predicted model IDs, shape (B,)
            probabilities: Softmax probabilities over models, shape (B, num_models)
        """
        with torch.no_grad():
            logits = self.forward(sample_texts, mode_ids)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        return predictions, probabilities

    def predict_top_k(
        self,
        sample_texts: List[str],
        mode_ids: torch.Tensor,
        k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k models for each sample.

        Args:
            sample_texts: List of sample text strings (length B)
            mode_ids: Mode IDs, shape (B,)
            k: Number of top models to return

        Returns:
            top_k_models: Top-k model IDs, shape (B, k)
            top_k_probs: Top-k probabilities, shape (B, k)
        """
        with torch.no_grad():
            logits = self.forward(sample_texts, mode_ids)
            probabilities = F.softmax(logits, dim=1)

            top_k_probs, top_k_models = torch.topk(probabilities, k=k, dim=1)

        return top_k_models, top_k_probs


def create_soft_labels_from_metrics(
    df_per_sample: "pd.DataFrame",
    metric_column: str = "reward_accuracy",
    temperature: float = 1.0,
    model_to_id: Dict[str, int] = None,
) -> torch.Tensor:
    """
    Create soft label distribution from metric values.

    For each (sample_id, mode_id), creates a probability distribution over models
    using softmax over metric values.

    Args:
        df_per_sample: DataFrame with columns [sample_id, mode_id, model_name, metric_column]
            Should contain one row per (sample, mode, model) combination
        metric_column: Column to use for soft labels (higher = better)
        temperature: Temperature for softmax (higher = softer distribution)
        model_to_id: Mapping from model_name to integer ID

    Returns:
        soft_labels: Tensor of shape (N, num_models) where N = unique (sample_id, mode_id) pairs
    """
    import pandas as pd
    import numpy as np

    if model_to_id is None:
        raise ValueError("model_to_id mapping is required")

    num_models = len(model_to_id)

    # Group by (sample_id, mode_id)
    grouped = df_per_sample.groupby(["sample_id", "mode_id"])

    soft_labels_list = []

    for (sample_id, mode_id), group in grouped:
        # Get metrics for all models
        model_metrics = np.zeros(num_models)

        for _, row in group.iterrows():
            model_name = row["model_name"]
            metric_value = row[metric_column]

            if model_name in model_to_id:
                model_id = model_to_id[model_name]
                model_metrics[model_id] = metric_value

        # Apply softmax with temperature
        exp_metrics = np.exp(model_metrics / temperature)
        soft_probs = exp_metrics / exp_metrics.sum()

        soft_labels_list.append(soft_probs)

    soft_labels = torch.tensor(np.array(soft_labels_list), dtype=torch.float32)

    logger.info(f"Created soft labels: shape {soft_labels.shape}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Entropy (avg): {-(soft_labels * torch.log(soft_labels + 1e-10)).sum(dim=1).mean():.3f}")

    return soft_labels
