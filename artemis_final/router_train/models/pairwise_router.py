"""
Pairwise ranking router model for VLM routing.

Learns to score (sample, model, mode) triples and predict which model
is preferred using pairwise comparisons.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class PairwiseRouterModel(nn.Module):
    """
    Pairwise ranking router that scores (sample, model, mode) triples.

    Architecture:
        1. Text encoder (DistilBERT) for sample text
        2. Learnable embeddings for model_id and mode_id
        3. MLP to combine and produce scalar score
        4. Training: Compare scores for (sample, model_i, mode) vs (sample, model_j, mode)
    """

    def __init__(
        self,
        num_models: int,
        num_modes: int = 4,
        text_encoder_name: str = "distilbert-base-uncased",
        model_embed_dim: int = 32,
        mode_embed_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_models: Number of VLM models
            num_modes: Number of routing modes (accuracy, cheap, fast, balanced)
            text_encoder_name: Hugging Face model name for text encoding
            model_embed_dim: Dimension of model embeddings
            mode_embed_dim: Dimension of mode embeddings
            hidden_dim: Hidden layer size in MLP
            dropout: Dropout probability
        """
        super().__init__()

        self.num_models = num_models
        self.num_modes = num_modes

        # Text encoder (shared with reward router)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size

        # Learnable embeddings
        self.model_embeddings = nn.Embedding(num_models, model_embed_dim)
        self.mode_embeddings = nn.Embedding(num_modes, mode_embed_dim)

        # MLP to combine and score
        combined_dim = text_dim + model_embed_dim + mode_embed_dim
        self.scorer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Scalar score
        )

        logger.info(f"Initialized PairwiseRouterModel:")
        logger.info(f"  Models: {num_models}, Modes: {num_modes}")
        logger.info(f"  Text encoder: {text_encoder_name} (dim={text_dim})")
        logger.info(f"  Model embed: {model_embed_dim}, Mode embed: {mode_embed_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}, Dropout: {dropout}")

    def forward(
        self,
        sample_texts: List[str],
        model_ids: torch.Tensor,
        mode_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to compute scores for (sample, model, mode) triples.

        Args:
            sample_texts: List of sample text strings (length B)
            model_ids: Model IDs, shape (B,)
            mode_ids: Mode IDs, shape (B,)

        Returns:
            scores: Scalar scores for each triple, shape (B,)
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

        # Embed model and mode
        model_emb = self.model_embeddings(model_ids.to(device))  # (B, model_embed_dim)
        mode_emb = self.mode_embeddings(mode_ids.to(device))  # (B, mode_embed_dim)

        # Combine and score
        combined = torch.cat([text_emb, model_emb, mode_emb], dim=1)  # (B, combined_dim)
        scores = self.scorer(combined).squeeze(-1)  # (B,)

        return scores

    def compute_pairwise_loss(
        self,
        sample_texts: List[str],
        model_i_ids: torch.Tensor,
        model_j_ids: torch.Tensor,
        mode_ids: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute margin ranking loss for pairwise examples.

        Loss = max(0, margin - (score_i - score_j))

        Args:
            sample_texts: List of sample text strings (length B)
            model_i_ids: IDs of preferred models, shape (B,)
            model_j_ids: IDs of less preferred models, shape (B,)
            mode_ids: Mode IDs, shape (B,)
            margin: Minimum score difference to enforce

        Returns:
            loss: Scalar tensor
        """
        # Compute scores for both models
        scores_i = self.forward(sample_texts, model_i_ids, mode_ids)  # (B,)
        scores_j = self.forward(sample_texts, model_j_ids, mode_ids)  # (B,)

        # Margin ranking loss: encourage score_i > score_j + margin
        loss = torch.clamp(margin - (scores_i - scores_j), min=0.0).mean()

        return loss

    def rank_models(
        self,
        sample_texts: List[str],
        model_ids: List[int],
        mode_id: int,
    ) -> List[int]:
        """
        Rank models for given samples and mode.

        Args:
            sample_texts: List of sample text strings (length N)
            model_ids: List of model IDs to rank (length M)
            mode_id: Mode ID (single value)

        Returns:
            ranked_model_ids: List of model IDs sorted by score (descending)
        """
        device = next(self.parameters()).device

        # Create all (sample, model, mode) combinations
        all_sample_texts = []
        all_model_ids = []
        all_mode_ids = []

        for sample_text in sample_texts:
            for model_id in model_ids:
                all_sample_texts.append(sample_text)
                all_model_ids.append(model_id)
                all_mode_ids.append(mode_id)

        all_model_ids = torch.tensor(all_model_ids, dtype=torch.long)
        all_mode_ids = torch.tensor(all_mode_ids, dtype=torch.long)

        # Compute scores
        with torch.no_grad():
            scores = self.forward(all_sample_texts, all_model_ids, all_mode_ids)

        # Reshape to (N_samples, M_models)
        scores = scores.view(len(sample_texts), len(model_ids))

        # Rank models (descending order)
        ranked_indices = torch.argsort(scores, dim=1, descending=True)

        # Convert to model IDs
        ranked_model_ids = []
        for i in range(len(sample_texts)):
            sample_ranking = [model_ids[idx] for idx in ranked_indices[i].tolist()]
            ranked_model_ids.append(sample_ranking)

        return ranked_model_ids

    def predict_best_model(
        self,
        sample_texts: List[str],
        model_ids: List[int],
        mode_id: int,
    ) -> List[int]:
        """
        Predict best model for each sample.

        Args:
            sample_texts: List of sample text strings (length N)
            model_ids: List of candidate model IDs
            mode_id: Mode ID

        Returns:
            best_model_ids: List of predicted best model ID for each sample (length N)
        """
        rankings = self.rank_models(sample_texts, model_ids, mode_id)
        return [ranking[0] for ranking in rankings]
