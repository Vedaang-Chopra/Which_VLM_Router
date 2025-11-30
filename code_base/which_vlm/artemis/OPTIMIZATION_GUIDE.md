# Router Training Optimization Guide

## Overview

This guide explains how to use the optimized pre-cached dataset pipeline for faster training.

## Problem Solved

**Original Issue**: The `RouterDataset` class fetches images from Cauldron on-the-fly during training, which means:
- Every epoch re-fetches the same images
- Network latency for HuggingFace streaming
- Slow training iterations
- Repeated decompression and processing

**Solution**: Pre-fetch and cache all images once, then use cached tensors during training.

---

## Step 1: Pre-fetch and Cache Images

Run the caching script **once** before training:

```bash
cd /storage/ice1/1/0/vchopra37/projects/vlm_router/code_base/which_vlm/artemis
python 00_prefetch_and_cache_dataset.py
```

This will:
1. Load all train/val/test datasets
2. Fetch images from Cauldron in parallel (32 workers by default)
3. Process them with CLIP image processor
4. Save to `cached_datasets/` directory

**Expected output:**
```
Processing train split (63,963 samples)
Fetching images with 32 parallel workers...
Fetching images: 100%|██████████| 63963/63963 [XX:XX<00:00, XXX.XX img/s]
✓ Cache saved (XXXX.X MB)
```

The cached files will be:
- `cached_datasets/cached_train.pt` (~X GB)
- `cached_datasets/cached_val.pt` (~X MB)
- `cached_datasets/cached_test.pt` (~X MB)

---

## Step 2: Modify Training Notebook

### Changes to `06_training_router.ipynb`:

#### Change 1: Import the cached dataset class

**Replace:**
```python
from imports.training_imports import RouterDataset
```

**With:**
```python
from imports.cached_dataset import CachedRouterDataset
```

#### Change 2: Update dataset initialization (Cell 18)

**Replace:**
```python
# Create datasets
print("\nCreating datasets...")
train_dataset = RouterDataset(
    train_df,
    config.image_root,
    image_processor,
    tokenizer,
    config.max_text_length,
    model_names,
    use_soft_labels=True,
    use_image=config.use_image
)

val_dataset = RouterDataset(
    val_df,
    config.image_root,
    image_processor,
    tokenizer,
    config.max_text_length,
    model_names,
    use_soft_labels=True,
    use_image=config.use_image
)

test_dataset = RouterDataset(
    test_df,
    config.image_root,
    image_processor,
    tokenizer,
    config.max_text_length,
    model_names,
    use_soft_labels=True,
    use_image=config.use_image
)
```

**With:**
```python
# Create datasets from cache
print("\nLoading cached datasets...")

cache_dir = Path.cwd() / "cached_datasets"

train_dataset = CachedRouterDataset(
    cache_file=cache_dir / "cached_train.pt",
    tokenizer=tokenizer,
    max_text_length=config.max_text_length,
    model_names=model_names,
    use_soft_labels=True,
)

val_dataset = CachedRouterDataset(
    cache_file=cache_dir / "cached_val.pt",
    tokenizer=tokenizer,
    max_text_length=config.max_text_length,
    model_names=model_names,
    use_soft_labels=True,
)

test_dataset = CachedRouterDataset(
    cache_file=cache_dir / "cached_test.pt",
    tokenizer=tokenizer,
    max_text_length=config.max_text_length,
    model_names=model_names,
    use_soft_labels=True,
)
```

#### Change 3: Remove image processor loading (Cell 18)

You can **remove or comment out** this section since images are pre-processed:

```python
# Load image processor and tokenizer
print("Loading image processor and tokenizer...")
image_processor = CLIPImageProcessor.from_pretrained(config.vision_encoder_name)  # <-- Can remove this line
tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer_name)
```

Keep only:
```python
# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer_name)
```

---

## Performance Improvements

### Before (Original):
- **First epoch**: Fetches all 63,963 images from Cauldron (streaming)
- **Every subsequent epoch**: Re-fetches all images again
- **Total time per epoch**: ~XX minutes (depending on network)

### After (Cached):
- **One-time setup**: Run caching script once (~XX minutes for all splits)
- **First epoch**: Loads images from disk cache (instant)
- **Every subsequent epoch**: Already in memory
- **Total time per epoch**: ~XX minutes (much faster)

### Expected Speedup:
- **Data loading**: 10-50x faster (depending on network)
- **Training iteration**: 2-5x faster overall
- **Total training time**: Reduced by 50-70%

---

## Configuration Options

### Parallel Workers

In `00_prefetch_and_cache_dataset.py`, adjust the number of workers:

```python
@dataclass
class CacheConfig:
    max_workers: int = 32  # Increase for faster caching (if you have enough cores)
```

Recommended values:
- **CPU with 16 cores**: 16-24 workers
- **CPU with 32 cores**: 32-48 workers
- **CPU with 64+ cores**: 48-64 workers

### Cache Location

By default, cached files are saved to:
```
code_base/which_vlm/artemis/cached_datasets/
```

To change the location, modify `cache_dir` in `CacheConfig`.

---

## Troubleshooting

### Issue: "Cache file already exists"

If you want to regenerate the cache:
```bash
rm -rf cached_datasets/
python 00_prefetch_and_cache_dataset.py
```

### Issue: "Failed to load image"

Some images may fail to fetch from Cauldron. The script will:
1. Retry 3 times with exponential backoff
2. Use a blank gray placeholder if all retries fail
3. Log the error for inspection

Check the error messages in the output to see which samples failed.

### Issue: Out of memory during caching

If you run out of RAM:
1. Reduce `max_workers` in `CacheConfig`
2. Process splits one at a time (comment out the others)
3. Use a machine with more RAM

### Issue: Disk space

The cached files require approximately:
- Train: ~X GB
- Val: ~X MB
- Test: ~X MB
- **Total: ~X GB**

Ensure you have enough disk space in the cache directory.

---

## Architecture Details

### Original Flow:
```
Training Loop
  └─> DataLoader
       └─> RouterDataset.__getitem__(idx)
            └─> fetch_cauldron_image()  # ← Slow! Network call every time
                 └─> Stream from HuggingFace
                 └─> Decode image
                 └─> Process with CLIP
            └─> Tokenize text
            └─> Return batch
```

### Optimized Flow:
```
One-time Setup:
  └─> 00_prefetch_and_cache_dataset.py
       └─> Parallel fetch all images
       └─> Process with CLIP
       └─> Save to disk

Training Loop:
  └─> DataLoader
       └─> CachedRouterDataset.__getitem__(idx)
            └─> Load pre-processed pixel_values[idx]  # ← Fast! Tensor indexing
            └─> Tokenize text (still on-the-fly, very fast)
            └─> Return batch
```

---

## Verification

After running the caching script, verify the cached files:

```python
import torch

# Load and inspect cached data
cache = torch.load("cached_datasets/cached_train.pt")

print(f"Pixel values shape: {cache['pixel_values'].shape}")
print(f"Dataframe shape: {cache['dataframe'].shape}")
print(f"Number of errors: {cache['config']['n_errors']}")
```

Expected output:
```
Pixel values shape: torch.Size([63963, 3, 224, 224])
Dataframe shape: (63963, XXX)
Number of errors: 0  (or small number)
```

---

## Next Steps

1. **Run the caching script** (one time):
   ```bash
   python 00_prefetch_and_cache_dataset.py
   ```

2. **Modify the training notebook** as described above

3. **Run training** and enjoy faster iterations!

4. **Optional**: For even faster training, consider:
   - Using larger batch sizes (now that data loading is faster)
   - Increasing number of workers in DataLoader
   - Using mixed precision training (if not already enabled)

---

## Files Created

- `00_prefetch_and_cache_dataset.py` - Caching script
- `imports/cached_dataset.py` - CachedRouterDataset class
- `OPTIMIZATION_GUIDE.md` - This guide
- `cached_datasets/` - Directory for cached data (created by script)

---

## Questions?

If you encounter any issues or have questions about the optimization, check:
1. Error messages from the caching script
2. Verify cache files exist and have correct sizes
3. Ensure enough disk space and RAM
4. Check that all imports are correct in the training notebook
