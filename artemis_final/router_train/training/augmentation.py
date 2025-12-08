"""
Shared metadata augmentation utilities for VLM router training.

Implements 70/20/10 augmentation strategy:
- 70% - Question + image metadata (general-purpose mode)
- 20% - Image metadata only (minimal context)
- 10% - Full metadata (task/dataset labels)
"""

import numpy as np
import pandas as pd
from typing import Optional


def build_augmented_input_text(
    row: pd.Series,
    enable_augmentation: bool = True,
    question_only_ratio: float = 0.70,
    image_metadata_only_ratio: float = 0.20,
    full_metadata_ratio: float = 0.10,
) -> str:
    """
    Build input text with metadata augmentation.

    Args:
        row: DataFrame row with columns: prompt_raw, txt_prompt_length_words,
             img_width, img_height, img_aspect_ratio, router_task, source_dataset, etc.
        enable_augmentation: If False, always use question+image (70% mode)
        question_only_ratio: Probability of question + image metadata (70%)
        image_metadata_only_ratio: Probability of image metadata only (20%)
        full_metadata_ratio: Probability of full metadata (10%)

    Returns:
        Formatted input text string matching [ROUTER] format
    """
    # Extract metadata
    prompt_raw = row.get('prompt_raw', '')
    prompt_len = row.get('txt_prompt_length_words', 0)
    img_width = row.get('img_width', 0)
    img_height = row.get('img_height', 0)
    img_ar = row.get('img_aspect_ratio', 0.0)
    router_task = row.get('router_task', '')
    source_dataset = row.get('source_dataset', '')
    source_config = row.get('source_config', '')
    data_split = row.get('data_split', '')

    # Augmentation logic
    if not enable_augmentation:
        # Default to 70% mode (question + image metadata)
        return (
            f"[ROUTER] PromptLenWords: {prompt_len}. "
            f"ImgWidth: {img_width}. ImgHeight: {img_height}. "
            f"ImgAR: {img_ar:.2f}. "
            f"Question: {prompt_raw}"
        )

    rand = np.random.random()

    if rand < question_only_ratio:  # 70% - Question + image metadata
        return (
            f"[ROUTER] PromptLenWords: {prompt_len}. "
            f"ImgWidth: {img_width}. ImgHeight: {img_height}. "
            f"ImgAR: {img_ar:.2f}. "
            f"Question: {prompt_raw}"
        )

    elif rand < (question_only_ratio + image_metadata_only_ratio):  # 20% - Image metadata only
        return (
            f"[ROUTER] ImgWidth: {img_width}. ImgHeight: {img_height}. "
            f"ImgAR: {img_ar:.2f}. "
            f"Question: {prompt_raw}"
        )

    else:  # 10% - Full metadata
        return (
            f"[ROUTER] Task: {router_task}. Dataset: {source_dataset}. "
            f"SourceConfig: {source_config}. Split: {data_split}. "
            f"PromptLenWords: {prompt_len}. "
            f"ImgWidth: {img_width}. ImgHeight: {img_height}. "
            f"ImgAR: {img_ar:.2f}. "
            f"Question: {prompt_raw}"
        )
