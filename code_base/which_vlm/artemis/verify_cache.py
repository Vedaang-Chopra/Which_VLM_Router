#!/usr/bin/env python3
"""
Quick verification script to check if cached datasets are properly created.

Usage:
    python verify_cache.py
"""

from pathlib import Path
import torch

def check_cache_file(cache_path: Path, expected_samples: int, split_name: str):
    """Verify a single cache file."""
    if not cache_path.exists():
        print(f"❌ {split_name}: Cache file NOT found at {cache_path}")
        return False

    try:
        # Load cache
        cache = torch.load(cache_path, map_location='cpu')

        # Check structure
        required_keys = ['pixel_values', 'dataframe', 'config']
        missing_keys = [k for k in required_keys if k not in cache]
        if missing_keys:
            print(f"❌ {split_name}: Missing keys in cache: {missing_keys}")
            return False

        # Check dimensions
        pixel_values = cache['pixel_values']
        df = cache['dataframe']

        if pixel_values.shape[0] != len(df):
            print(f"❌ {split_name}: Shape mismatch! pixel_values={pixel_values.shape[0]}, df={len(df)}")
            return False

        if pixel_values.shape[0] != expected_samples:
            print(f"⚠️  {split_name}: Sample count mismatch! Expected {expected_samples}, got {pixel_values.shape[0]}")

        # File size
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        n_errors = cache['config'].get('n_errors', 0)

        print(f"✅ {split_name}:")
        print(f"   File: {cache_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Samples: {len(df):,}")
        print(f"   Pixel values: {pixel_values.shape}")
        print(f"   Errors: {n_errors}")

        return True

    except Exception as e:
        print(f"❌ {split_name}: Failed to load cache: {e}")
        return False


def main():
    print("="*60)
    print("VERIFYING CACHED DATASETS")
    print("="*60 + "\n")

    cache_dir = Path(__file__).parent / "cached_datasets"

    if not cache_dir.exists():
        print(f"❌ Cache directory NOT found: {cache_dir}")
        print("\n📝 To create cache, run:")
        print("   python 00_prefetch_and_cache_dataset.py")
        return

    print(f"Cache directory: {cache_dir}\n")

    # Expected sizes (from the parquet files)
    checks = [
        ("train", "cached_train.pt", 63963),
        ("val", "cached_val.pt", 13706),
        ("test", "cached_test.pt", 13707),
    ]

    all_ok = True
    for split_name, filename, expected_samples in checks:
        cache_path = cache_dir / filename
        ok = check_cache_file(cache_path, expected_samples, split_name.upper())
        all_ok = all_ok and ok
        print()

    print("="*60)
    if all_ok:
        print("✅ ALL CACHE FILES VERIFIED!")
        print("\n📝 Ready to train! Use the cached datasets in your training notebook.")
        print("   See TRAINING_NOTEBOOK_CHANGES.md for instructions.")
    else:
        print("❌ SOME CACHE FILES HAVE ISSUES")
        print("\n📝 To regenerate cache, run:")
        print("   python 00_prefetch_and_cache_dataset.py")
    print("="*60)


if __name__ == "__main__":
    main()
