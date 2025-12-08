"""
PyTorch dataset and dataloader for router training.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import Config, RouterModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


class RewardRouterDataset(Dataset):
    """
    PyTorch dataset for reward-based router training.

    Each item represents a (sample, model, mode) tuple with:
        - Tokenized text input (query + metadata)
        - Model ID
        - Mode ID
        - Target reward
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 256,
        split: str = "train",
        enable_augmentation: bool = True,
        question_only_ratio: float = 0.70,
        image_metadata_only_ratio: float = 0.20,
        full_metadata_ratio: float = 0.10,
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with router training data
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum sequence length for tokenization
            split: Split name for logging ("train", "val", "test")
            enable_augmentation: Enable metadata masking augmentation (only for train split)
            question_only_ratio: Ratio of samples with question + image metadata only (no task/dataset)
            image_metadata_only_ratio: Ratio of samples with minimal image metadata only
            full_metadata_ratio: Ratio of samples with full metadata
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split

        # Augmentation config (only apply to train split)
        self.enable_augmentation = enable_augmentation and (split == "train")
        self.question_only_ratio = question_only_ratio
        self.image_metadata_only_ratio = image_metadata_only_ratio
        self.full_metadata_ratio = full_metadata_ratio

        # Validate ratios sum to 1.0
        total = question_only_ratio + image_metadata_only_ratio + full_metadata_ratio
        assert abs(total - 1.0) < 1e-6, f"Augmentation ratios must sum to 1.0 (got {total})"

        logger.info(f"Created {split} dataset with {len(self.df)} samples")
        if self.enable_augmentation:
            logger.info(f"  Metadata augmentation enabled:")
            logger.info(f"    Question+Image: {question_only_ratio:.0%}")
            logger.info(f"    Image only: {image_metadata_only_ratio:.0%}")
            logger.info(f"    Full metadata: {full_metadata_ratio:.0%}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index

        Returns:
            Dictionary with keys:
                - input_ids: Token IDs [seq_len]
                - attention_mask: Attention mask [seq_len]
                - model_id: Model ID (scalar)
                - mode_id: Mode ID (scalar)
                - reward: Target reward (scalar)
                - sample_id: Sample ID (string, for debugging)
        """
        row = self.df.iloc[idx]

        # Build input text
        input_text = self._build_input_text(row)

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "model_id": torch.tensor(row["model_id"], dtype=torch.long),
            "mode_id": torch.tensor(row["mode_id"], dtype=torch.long),
            "reward": torch.tensor(row["reward"], dtype=torch.float),
            "sample_id": row["sample_id"],
        }

    def _build_input_text(self, row: pd.Series) -> str:
        """
        Build input text from row data with optional metadata masking.

        Augmentation strategy (only for train split when enabled):
        - 70%: Question + Image metadata only (no task/dataset labels)
          Format: [ROUTER] PromptLenWords: X. ImgWidth: W. ImgHeight: H. ImgAR: A. Question: {prompt}
        - 20%: Image metadata only (minimal)
          Format: [ROUTER] ImgWidth: W. ImgHeight: H. ImgAR: A. Question: {prompt}
        - 10%: Full metadata (original format)
          Format: [ROUTER] Task: T. Dataset: D. SourceConfig: C. Split: S. PromptLenWords: X.
                  ImgWidth: W. ImgHeight: H. ImgAR: A. Question: {prompt}

        For val/test splits or when augmentation disabled, always use full metadata.

        Args:
            row: DataFrame row

        Returns:
            Formatted input string
        """
        # Get values with defaults
        router_task = row.get("router_task", "unknown")
        source_dataset = row.get("source_dataset", "unknown")
        source_config = row.get("source_config", "unknown")
        data_split = row.get("data_split", "unknown")

        # Handle both column name variations
        prompt_len_words = int(row.get("txt_prompt_length_words", row.get("prompt_len_words", 0)))

        img_width = int(row.get("img_width", 0))
        img_height = int(row.get("img_height", 0))
        img_aspect_ratio = float(row.get("img_aspect_ratio", 1.0))
        prompt_raw = str(row.get("prompt_raw", ""))

        # If augmentation disabled or not train split, use full metadata
        if not self.enable_augmentation:
            return (
                f"[ROUTER] Task: {router_task}. Dataset: {source_dataset}. "
                f"SourceConfig: {source_config}. Split: {data_split}. "
                f"PromptLenWords: {prompt_len_words}. "
                f"ImgWidth: {img_width}. ImgHeight: {img_height}. "
                f"ImgAR: {img_aspect_ratio:.2f}. "
                f"Question: {prompt_raw}"
            )

        # Apply metadata masking augmentation
        rand = np.random.random()

        if rand < self.question_only_ratio:
            # 70%: Question + Image metadata only (NO task/dataset labels)
            # This is the primary inference format
            return (
                f"[ROUTER] PromptLenWords: {prompt_len_words}. "
                f"ImgWidth: {img_width}. ImgHeight: {img_height}. "
                f"ImgAR: {img_aspect_ratio:.2f}. "
                f"Question: {prompt_raw}"
            )

        elif rand < (self.question_only_ratio + self.image_metadata_only_ratio):
            # 20%: Image metadata only (minimal format)
            # Teaches model to work with minimal context
            return (
                f"[ROUTER] ImgWidth: {img_width}. ImgHeight: {img_height}. "
                f"ImgAR: {img_aspect_ratio:.2f}. "
                f"Question: {prompt_raw}"
            )

        else:
            # 10%: Full metadata (original format)
            # Maintains performance on known datasets
            return (
                f"[ROUTER] Task: {router_task}. Dataset: {source_dataset}. "
                f"SourceConfig: {source_config}. Split: {data_split}. "
                f"PromptLenWords: {prompt_len_words}. "
                f"ImgWidth: {img_width}. ImgHeight: {img_height}. "
                f"ImgAR: {img_aspect_ratio:.2f}. "
                f"Question: {prompt_raw}"
            )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of dataset items

    Returns:
        Batched dictionary
    """
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    model_id = torch.stack([item["model_id"] for item in batch])
    mode_id = torch.stack([item["mode_id"] for item in batch])
    reward = torch.stack([item["reward"] for item in batch])

    # Keep sample_ids as list
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "model_id": model_id,
        "mode_id": mode_id,
        "reward": reward,
        "sample_ids": sample_ids,
    }


def split_by_sample(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by sample_id to prevent leakage.

    Each sample_id appears in only one split, but all (model, mode) combinations
    for that sample are kept together.

    Args:
        df: Input dataframe
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Get unique sample IDs
    unique_samples = df["sample_id"].unique()
    n_samples = len(unique_samples)

    # Shuffle
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_samples)

    # Compute split indices
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    # Split sample IDs
    train_samples = set(unique_samples[:train_end])
    val_samples = set(unique_samples[train_end:val_end])
    test_samples = set(unique_samples[val_end:])

    # Create split dataframes
    train_df = df[df["sample_id"].isin(train_samples)].copy()
    val_df = df[df["sample_id"].isin(val_samples)].copy()
    test_df = df[df["sample_id"].isin(test_samples)].copy()

    logger.info(f"Dataset split by sample_id (seed={seed}):")
    logger.info(f"  Train: {len(train_samples)} samples, {len(train_df)} rows")
    logger.info(f"  Val:   {len(val_samples)} samples, {len(val_df)} rows")
    logger.info(f"  Test:  {len(test_samples)} samples, {len(test_df)} rows")

    # Verify no leakage
    assert len(train_samples & val_samples) == 0, "Train/val sample overlap!"
    assert len(train_samples & test_samples) == 0, "Train/test sample overlap!"
    assert len(val_samples & test_samples) == 0, "Val/test sample overlap!"

    return train_df, val_df, test_df


def create_dataloaders(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    config: TrainingConfig,
    model_config: RouterModelConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test dataloaders.

    Args:
        df: Full dataset dataframe
        tokenizer: HuggingFace tokenizer
        config: Training configuration
        model_config: Model configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_df, val_df, test_df)
    """
    logger.info("Creating dataloaders...")

    # Split data
    train_df, val_df, test_df = split_by_sample(
        df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    # Create datasets
    train_dataset = RewardRouterDataset(
        train_df,
        tokenizer,
        max_seq_length=model_config.max_seq_length,
        split="train",
    )

    val_dataset = RewardRouterDataset(
        val_df,
        tokenizer,
        max_seq_length=model_config.max_seq_length,
        split="val",
    )

    test_dataset = RewardRouterDataset(
        test_df,
        tokenizer,
        max_seq_length=model_config.max_seq_length,
        split="test",
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )

    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader, train_df, val_df, test_df


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load router dataset from parquet file.

    Args:
        dataset_path: Path to parquet file

    Returns:
        Dataset dataframe
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def split_by_data_split_column(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset using pre-existing data_split column.

    Assumes df has a "data_split" column with values like "train", "val", "test".

    Args:
        df: Input dataframe with data_split column

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if "data_split" not in df.columns:
        raise ValueError("DataFrame must have 'data_split' column")

    # Split by data_split values
    train_df = df[df["data_split"] == "train"].copy()
    val_df = df[df["data_split"] == "val"].copy()
    test_df = df[df["data_split"] == "test"].copy()

    logger.info(f"Dataset split by data_split column:")
    logger.info(f"  Train: {len(train_df)} rows ({train_df['sample_id'].nunique()} unique samples)")
    logger.info(f"  Val:   {len(val_df)} rows ({val_df['sample_id'].nunique()} unique samples)")
    logger.info(f"  Test:  {len(test_df)} rows ({test_df['sample_id'].nunique()} unique samples)")

    return train_df, val_df, test_df


def load_reward_dataset(
    parquet_path: str,
    model_index_path: str,
    mode_index_path: str,
    split_by: str = "data_split",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load reward dataset and split into train/val/test.

    Args:
        parquet_path: Path to dataset parquet file
        model_index_path: Path to model index JSON (not used, for API compatibility)
        mode_index_path: Path to mode index JSON (not used, for API compatibility)
        split_by: How to split data ("data_split" uses existing column, "sample_id" does random split)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Loading reward dataset from: {parquet_path}")
    df = load_dataset(parquet_path)

    if split_by == "data_split":
        logger.info("Splitting by data_split column...")
        return split_by_data_split_column(df)
    elif split_by == "sample_id":
        logger.info("Splitting randomly by sample_id...")
        return split_by_sample(df)
    else:
        raise ValueError(f"Unknown split_by method: {split_by}")


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_seq_length: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    enable_augmentation: bool = True,
    question_only_ratio: float = 0.70,
    image_metadata_only_ratio: float = 0.20,
    full_metadata_ratio: float = 0.10,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build dataloaders from pre-split dataframes.

    This is a simpler API for notebook usage.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_seq_length: Maximum sequence length
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        enable_augmentation: Enable metadata masking augmentation (train only)
        question_only_ratio: Ratio of question+image metadata samples
        image_metadata_only_ratio: Ratio of minimal image metadata samples
        full_metadata_ratio: Ratio of full metadata samples

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Building dataloaders...")

    # Create datasets
    train_dataset = RewardRouterDataset(
        train_df,
        tokenizer,
        max_seq_length=max_seq_length,
        split="train",
        enable_augmentation=enable_augmentation,
        question_only_ratio=question_only_ratio,
        image_metadata_only_ratio=image_metadata_only_ratio,
        full_metadata_ratio=full_metadata_ratio,
    )

    val_dataset = RewardRouterDataset(
        val_df,
        tokenizer,
        max_seq_length=max_seq_length,
        split="val",
        enable_augmentation=False,  # Never augment validation
    )

    test_dataset = RewardRouterDataset(
        test_df,
        tokenizer,
        max_seq_length=max_seq_length,
        split="test",
        enable_augmentation=False,  # Never augment test
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
