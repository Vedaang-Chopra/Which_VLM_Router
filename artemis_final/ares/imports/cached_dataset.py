"""
Cached dataset loader for fast training.

Uses pre-fetched and pre-processed images from disk cache.
"""

import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd
from pathlib import Path


class CachedRouterDataset(Dataset):
    """Dataset that loads pre-cached images from disk."""

    def __init__(
        self,
        cache_file: Path,
        tokenizer,
        max_text_length: int = 256,
        model_names: List[str] = None,
        use_soft_labels: bool = True,
    ):
        """
        Args:
            cache_file: Path to the cached .pt file
            tokenizer: Tokenizer for text processing
            max_text_length: Max tokens for text
            model_names: List of model names for soft labels
            use_soft_labels: Whether to use soft labels
        """
        print(f"Loading cached dataset from: {cache_file}")

        # Load cache
        cache_data = torch.load(cache_file, map_location='cpu')

        self.pixel_values = cache_data['pixel_values']  # [N, 3, H, W]
        self.df = cache_data['dataframe']  # DataFrame
        self.error_messages = cache_data.get('error_messages', [])

        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.model_names = model_names
        self.use_soft_labels = use_soft_labels

        print(f"  Loaded {len(self.df):,} samples")
        print(f"  Pixel values shape: {self.pixel_values.shape}")

        n_errors = sum(1 for e in self.error_messages if e is not None)
        if n_errors > 0:
            print(f"  ⚠️  {n_errors} samples had image loading errors (using placeholders)")

    def __len__(self):
        return len(self.df)

    def _construct_router_text(self, row) -> str:
        """Construct the text input for the router."""
        parts = []

        # Task information
        if pd.notna(row.get('router_task')):
            parts.append(f"Task: {row['router_task']}")

        # Dataset information
        if pd.notna(row.get('source_dataset')):
            parts.append(f"Dataset: {row['source_dataset']}")

        # Question type
        if pd.notna(row.get('txt_question_type')):
            parts.append(f"QType: {row['txt_question_type']}")

        # Has multiple choice
        if pd.notna(row.get('txt_has_mc_options')):
            has_mc = "Yes" if row['txt_has_mc_options'] else "No"
            parts.append(f"HasMC: {has_mc}")

        # Image dimensions
        if pd.notna(row.get('img_width')) and pd.notna(row.get('img_height')):
            parts.append(f"ImgSize: {int(row['img_width'])}x{int(row['img_height'])}")

        # Aspect ratio
        if pd.notna(row.get('img_aspect_ratio')):
            parts.append(f"AR: {row['img_aspect_ratio']:.2f}")

        # Prompt length
        if pd.notna(row.get('txt_prompt_length_words')):
            parts.append(f"PromptWords: {int(row['txt_prompt_length_words'])}")

        # Add the actual prompt
        if pd.notna(row.get('prompt_raw')):
            parts.append(f"Prompt: {row['prompt_raw']}")

        return " | ".join(parts)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -----------------------------
        # 1. Get pre-cached pixel values
        # -----------------------------
        pixel_values = self.pixel_values[idx]  # Already a tensor [3, H, W]

        # -----------------------------
        # 2. Construct and tokenize text
        # -----------------------------
        router_text = self._construct_router_text(row)
        text_encoding = self.tokenizer(
            router_text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # -----------------------------
        # 3. Hard label
        # -----------------------------
        hard_label = row["router_best_model_id"]

        # -----------------------------
        # 4. Soft labels
        # -----------------------------
        soft_labels = torch.zeros(len(self.model_names), dtype=torch.float32)
        if self.use_soft_labels:
            for i, model_name in enumerate(self.model_names):
                col = f"router_soft_p_{model_name}"
                if col in row.index and pd.notna(row[col]):
                    soft_labels[i] = row[col]
            # Normalize to ensure sum=1
            if soft_labels.sum() > 0:
                soft_labels = soft_labels / soft_labels.sum()
            else:
                # Fallback to one-hot if soft labels missing
                soft_labels[hard_label] = 1.0
        else:
            soft_labels[hard_label] = 1.0

        return {
            "pixel_values": pixel_values,
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "hard_label": torch.tensor(hard_label, dtype=torch.long),
            "soft_labels": soft_labels,
            "sample_id": row.get("sample_id", f"sample_{idx}"),
        }


print("CachedRouterDataset class defined.")
