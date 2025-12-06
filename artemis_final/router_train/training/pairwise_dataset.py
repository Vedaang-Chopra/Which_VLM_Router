"""
Pairwise dataset for training ranking-based VLM router.

Generates (sample, model_i, model_j) pairs with preference labels based on
reward/metric comparisons.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def generate_pairwise_examples(
    df: pd.DataFrame,
    metric_column: str = "reward_accuracy",
    min_margin: float = 0.0,
    max_pairs_per_sample: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate pairwise examples from per-(sample, model) data.

    For each (sample_id, mode_id), creates pairs (model_i, model_j) where
    model_i is preferred over model_j based on metric_column.

    Args:
        df: DataFrame with columns [sample_id, model_name, mode_id, metric_column, ...]
        metric_column: Column to use for preference (higher = better)
        min_margin: Minimum metric difference to include pair (filters noise)
        max_pairs_per_sample: Max pairs per (sample_id, mode_id), None = all pairs

    Returns:
        DataFrame with columns:
            - sample_id: Sample identifier
            - mode_id: Mode identifier
            - model_i: Preferred model
            - model_j: Less preferred model
            - metric_i: Metric value for model_i
            - metric_j: Metric value for model_j
            - margin: metric_i - metric_j (always > 0)
            - prompt_raw: Prompt text (copied from original)
            - txt_prompt_length_chars, txt_prompt_length_words: Prompt metadata
            - img_width, img_height, img_aspect_ratio: Image metadata
    """
    required_cols = ["sample_id", "model_name", "mode_id", metric_column]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    pairs_list = []

    # Group by (sample_id, mode_id)
    grouped = df.groupby(["sample_id", "mode_id"])

    for (sample_id, mode_id), group in grouped:
        # Get all models for this (sample, mode)
        models = group[["model_name", metric_column]].values

        if len(models) < 2:
            continue  # Need at least 2 models to compare

        # Generate all pairs where model_i > model_j
        group_pairs = []
        for i in range(len(models)):
            for j in range(len(models)):
                if i == j:
                    continue

                model_i, metric_i = models[i]
                model_j, metric_j = models[j]

                # Only keep pairs where i is better than j
                if metric_i > metric_j:
                    margin = metric_i - metric_j
                    if margin >= min_margin:
                        group_pairs.append({
                            "sample_id": sample_id,
                            "mode_id": mode_id,
                            "model_i": model_i,
                            "model_j": model_j,
                            "metric_i": metric_i,
                            "metric_j": metric_j,
                            "margin": margin,
                        })

        # Limit pairs per sample if specified
        if max_pairs_per_sample is not None and len(group_pairs) > max_pairs_per_sample:
            # Sort by margin descending (keep most informative pairs)
            group_pairs = sorted(group_pairs, key=lambda x: x["margin"], reverse=True)
            group_pairs = group_pairs[:max_pairs_per_sample]

        pairs_list.extend(group_pairs)

    if len(pairs_list) == 0:
        raise ValueError("No valid pairs generated. Check metric_column and min_margin.")

    pairs_df = pd.DataFrame(pairs_list)

    # Copy metadata from original df (use first occurrence of each sample_id)
    metadata_cols = [
        "prompt_raw",
        "txt_prompt_length_chars",
        "txt_prompt_length_words",
        "img_width",
        "img_height",
        "img_aspect_ratio",
        "source_dataset",
        "router_task",
        "data_split",
    ]
    available_metadata = [col for col in metadata_cols if col in df.columns]

    sample_metadata = df.groupby("sample_id")[available_metadata].first().reset_index()
    pairs_df = pairs_df.merge(sample_metadata, on="sample_id", how="left")

    logger.info(f"Generated {len(pairs_df)} pairwise examples")
    logger.info(f"  Unique samples: {pairs_df['sample_id'].nunique()}")
    logger.info(f"  Pairs per sample: {len(pairs_df) / pairs_df['sample_id'].nunique():.1f} avg")
    logger.info(f"  Margin range: [{pairs_df['margin'].min():.3f}, {pairs_df['margin'].max():.3f}]")

    return pairs_df


class PairwiseRouterDataset(Dataset):
    """
    PyTorch dataset for pairwise ranking training.

    Each example is (sample_text, model_i_id, model_j_id) with label = 1.0
    (indicating model_i is preferred over model_j).
    """

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        model_to_id: Dict[str, int],
        mode_to_id: Dict[str, int],
    ):
        """
        Args:
            pairs_df: DataFrame from generate_pairwise_examples()
            model_to_id: Mapping from model_name to integer ID
            mode_to_id: Mapping from mode_id to integer ID
        """
        self.df = pairs_df.reset_index(drop=True)
        self.model_to_id = model_to_id
        self.mode_to_id = mode_to_id

        # Validate all models and modes are in mappings
        unknown_models = set(self.df["model_i"].unique()) | set(self.df["model_j"].unique())
        unknown_models = unknown_models - set(model_to_id.keys())
        if unknown_models:
            raise ValueError(f"Unknown models in pairs_df: {unknown_models}")

        unknown_modes = set(self.df["mode_id"].unique()) - set(mode_to_id.keys())
        if unknown_modes:
            raise ValueError(f"Unknown modes in pairs_df: {unknown_modes}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, int, int, int, float]:
        """
        Returns:
            sample_text: Input text (prompt + metadata)
            model_i_id: ID of preferred model
            model_j_id: ID of less preferred model
            mode_id: Mode ID
            label: Always 1.0 (model_i > model_j)
        """
        row = self.df.iloc[idx]

        # Build input text (same format as reward router)
        sample_text = self._build_input_text(row)

        model_i_id = self.model_to_id[row["model_i"]]
        model_j_id = self.model_to_id[row["model_j"]]
        mode_id = self.mode_to_id[row["mode_id"]]

        # Label = 1.0 means model_i is preferred
        label = 1.0

        return sample_text, model_i_id, model_j_id, mode_id, label

    def _build_input_text(self, row: pd.Series) -> str:
        """Build input text from sample metadata."""
        parts = [f"Prompt: {row['prompt_raw']}"]

        if pd.notna(row.get("txt_prompt_length_chars")):
            parts.append(f"PromptLen: {int(row['txt_prompt_length_chars'])} chars")

        if pd.notna(row.get("img_width")) and pd.notna(row.get("img_height")):
            parts.append(f"Image: {int(row['img_width'])}x{int(row['img_height'])}")

        if pd.notna(row.get("source_dataset")):
            parts.append(f"Dataset: {row['source_dataset']}")

        if pd.notna(row.get("router_task")):
            parts.append(f"Task: {row['router_task']}")

        return " | ".join(parts)


def collate_pairwise_batch(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
    """
    Collate function for pairwise dataloader.

    Args:
        batch: List of (sample_text, model_i_id, model_j_id, mode_id, label) tuples

    Returns:
        Dictionary with:
            - sample_texts: List[str]
            - model_i_ids: Tensor of shape (batch_size,)
            - model_j_ids: Tensor of shape (batch_size,)
            - mode_ids: Tensor of shape (batch_size,)
            - labels: Tensor of shape (batch_size,)
    """
    sample_texts, model_i_ids, model_j_ids, mode_ids, labels = zip(*batch)

    return {
        "sample_texts": list(sample_texts),
        "model_i_ids": torch.tensor(model_i_ids, dtype=torch.long),
        "model_j_ids": torch.tensor(model_j_ids, dtype=torch.long),
        "mode_ids": torch.tensor(mode_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }
