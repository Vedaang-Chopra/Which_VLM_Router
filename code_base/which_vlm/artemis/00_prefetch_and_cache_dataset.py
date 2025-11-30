"""
00 Pre-fetch and Cache Dataset for Fast Training

Purpose: Pre-fetch all images from Cauldron and create cached datasets

This script:
1. Loads all splits (train, val, test) of the router dataset
2. Parallel image fetching from Cauldron with caching
3. Pre-processes images with CLIP processor
4. Saves processed datasets to disk for instant loading during training

Benefits:
- Fetch images from Cauldron ONCE instead of on every epoch
- Parallel processing for speed
- Cached on disk for multiple training runs
- Progress tracking with tqdm
- Error handling and retries
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from PIL import Image

from transformers import CLIPImageProcessor

# Local imports
sys.path.append(str(Path(__file__).parent.parent / "dataset_builder"))
from check_data_utils import fetch_cauldron_image

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


@dataclass
class CacheConfig:
    """Configuration for dataset caching."""

    # Paths
    data_root: Path = Path(__file__).resolve().parent.parent.parent.parent.parent / "dataset" / "final_dataset"
    image_root: Path = Path(__file__).resolve().parent.parent.parent.parent.parent / "dataset" / "which_vlm_data" / "images"
    cache_dir: Path = Path(__file__).resolve().parent / "cached_datasets"

    # Vision encoder for preprocessing
    vision_encoder_name: str = "openai/clip-vit-base-patch32"

    # Parallel processing
    max_workers: int = 32  # Number of parallel threads for image fetching
    batch_size: int = 100  # Process images in batches

    # Error handling
    max_retries: int = 3

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_root.mkdir(parents=True, exist_ok=True)


def fetch_and_process_image(
    row_data: Tuple[int, pd.Series],
    image_root: Path,
    image_processor,
    max_retries: int = 3,
) -> Tuple[int, Optional[torch.Tensor], Optional[str]]:
    """
    Fetch image from Cauldron and process it.

    Returns:
        (index, pixel_values_tensor, error_message)
    """
    idx, row = row_data

    for attempt in range(max_retries):
        try:
            # Validate required fields
            if pd.isna(row.get('source_config')) or pd.isna(row.get('source_index')):
                raise ValueError(f"Missing source_config or source_index for row {idx}")

            # Fetch image from Cauldron (with caching)
            # fetch_cauldron_image will:
            # 1. Check local cache first (using image_hash)
            # 2. Stream from HuggingFace if not cached
            # 3. Save to cache for future use
            img, _ = fetch_cauldron_image(
                source_config=str(row['source_config']),
                source_index=int(row['source_index']),
                image_hash=row.get('image_bytes_hash', None) if pd.notna(row.get('image_bytes_hash')) else None,
                prefer_local_cache=True,
                image_root=image_root,
            )

            # Ensure RGB mode
            img = img.convert("RGB")

            # Process with CLIP processor
            pixel_values = image_processor(
                images=img,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)

            return idx, pixel_values, None

        except Exception as e:
            if attempt < max_retries - 1:
                # Retry with exponential backoff
                time.sleep(0.5 * (attempt + 1))
                continue
            else:
                # Last attempt failed
                error_msg = f"Failed after {max_retries} attempts: {str(e)[:100]}"
                print(f"⚠️  Sample {idx}: {error_msg}")
                # Return a blank image
                blank = Image.new("RGB", (224, 224), color="gray")
                pixel_values = image_processor(
                    images=blank,
                    return_tensors="pt",
                )["pixel_values"].squeeze(0)
                return idx, pixel_values, error_msg

    return idx, None, "Unknown error"


def fetch_images_parallel(
    df: pd.DataFrame,
    image_root: Path,
    image_processor,
    max_workers: int = 32,
    max_retries: int = 3,
) -> Tuple[List[torch.Tensor], List[Optional[str]]]:
    """
    Fetch and process all images in parallel.

    Returns:
        (pixel_values_list, error_messages_list)
    """
    n_samples = len(df)
    pixel_values_list = [None] * n_samples
    error_messages = [None] * n_samples

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                fetch_and_process_image,
                (idx, row),
                image_root,
                image_processor,
                max_retries,
            ): idx
            for idx, row in df.iterrows()
        }

        # Process completed tasks with progress bar
        with tqdm(total=n_samples, desc="Fetching images", unit="img") as pbar:
            for future in as_completed(futures):
                idx, pixel_values, error_msg = future.result()
                pixel_values_list[idx] = pixel_values
                error_messages[idx] = error_msg
                pbar.update(1)

    return pixel_values_list, error_messages


def process_and_cache_split(
    df: pd.DataFrame,
    split_name: str,
    config: CacheConfig,
    image_processor,
) -> Path:
    """
    Process a dataset split and cache it to disk.

    Returns:
        Path to cached file
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split ({len(df):,} samples)")
    print(f"{'='*60}\n")

    # Check if already cached
    cache_file = config.cache_dir / f"cached_{split_name}.pt"
    if cache_file.exists():
        print(f"✓ Cache file already exists: {cache_file}")
        print("  To regenerate, delete the file and re-run this script.")
        return cache_file

    # Fetch and process images in parallel
    print(f"Fetching images with {config.max_workers} parallel workers...")
    pixel_values_list, error_messages = fetch_images_parallel(
        df.reset_index(drop=True),  # Reset index for clean indexing
        config.image_root,
        image_processor,
        max_workers=config.max_workers,
        max_retries=config.max_retries,
    )

    # Report errors
    n_errors = sum(1 for e in error_messages if e is not None)
    if n_errors > 0:
        print(f"\n⚠️  {n_errors} images failed to load (using blank placeholders)")
        # Show first few errors
        error_count = 0
        for i, err in enumerate(error_messages):
            if err and error_count < 5:
                print(f"  Sample {i}: {err}")
                error_count += 1

    # Stack all pixel values into a single tensor
    print(f"\nStacking {len(pixel_values_list)} image tensors...")
    pixel_values_tensor = torch.stack(pixel_values_list)

    # Prepare cache data
    cache_data = {
        'pixel_values': pixel_values_tensor,  # [N, 3, H, W]
        'dataframe': df.reset_index(drop=True),  # Original dataframe
        'error_messages': error_messages,
        'config': {
            'vision_encoder_name': config.vision_encoder_name,
            'n_samples': len(df),
            'n_errors': n_errors,
        }
    }

    # Save to disk
    print(f"\nSaving cache to: {cache_file}")
    torch.save(cache_data, cache_file)

    # Report file size
    file_size_mb = cache_file.stat().st_size / (1024 * 1024)
    print(f"✓ Cache saved ({file_size_mb:.1f} MB)")

    return cache_file


def main():
    """Main execution function."""

    # Initialize config
    config = CacheConfig()

    print("\n=== Cache Configuration ===")
    print(f"Data root:       {config.data_root}")
    print(f"Image cache:     {config.image_root}")
    print(f"Output cache:    {config.cache_dir}")
    print(f"Max workers:     {config.max_workers}")
    print(f"Batch size:      {config.batch_size}")

    # Load datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60 + "\n")

    train_df = pd.read_parquet(config.data_root / "router_final" / "router_train_final.parquet")
    val_df = pd.read_parquet(config.data_root / "router_final" / "router_val_final.parquet")
    test_df = pd.read_parquet(config.data_root / "router_final" / "router_test_final.parquet")

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df):,} samples")

    # Initialize image processor
    print(f"\nLoading image processor: {config.vision_encoder_name}")
    image_processor = CLIPImageProcessor.from_pretrained(config.vision_encoder_name)
    print("✓ Image processor loaded")

    # Process each split
    cache_files = {}

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        cache_file = process_and_cache_split(df, split_name, config, image_processor)
        cache_files[split_name] = cache_file

    # Verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60 + "\n")

    for split_name, cache_file in cache_files.items():
        print(f"\n{split_name.upper()}:")

        # Load cache
        cache_data = torch.load(cache_file, map_location='cpu')

        # Check contents
        pixel_values = cache_data['pixel_values']
        df = cache_data['dataframe']
        cfg = cache_data['config']

        print(f"  File: {cache_file}")
        print(f"  File size: {cache_file.stat().st_size / (1024**2):.1f} MB")
        print(f"  Pixel values shape: {pixel_values.shape}")
        print(f"  Dataframe shape: {df.shape}")
        print(f"  Vision encoder: {cfg['vision_encoder_name']}")
        print(f"  Samples: {cfg['n_samples']:,}")
        print(f"  Errors: {cfg['n_errors']}")

        # Verify shape consistency
        assert pixel_values.shape[0] == len(df), f"Shape mismatch in {split_name}"
        print(f"  ✓ Shape consistency verified")

    # Summary
    print("\n" + "="*60)
    print("CACHING COMPLETE!")
    print("="*60 + "\n")

    print("Cached datasets are ready for training at:")
    print(f"  {config.cache_dir}")
    print("\nCached files:")
    for f in sorted(config.cache_dir.glob("*.pt")):
        size_mb = f.stat().st_size / (1024**2)
        print(f"  - {f.name} ({size_mb:.1f} MB)")

    total_size = sum(f.stat().st_size for f in config.cache_dir.glob("*.pt")) / (1024**2)
    print(f"\nTotal cache size: {total_size:.1f} MB")

    print("\n" + "="*60)
    print("Next step: Run the modified training notebook!")
    print("="*60)


if __name__ == "__main__":
    main()
