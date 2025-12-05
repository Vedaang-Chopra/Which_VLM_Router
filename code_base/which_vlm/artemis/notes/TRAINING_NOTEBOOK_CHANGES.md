# Exact Changes for 06_training_router.ipynb

## Summary

Replace the on-the-fly image fetching with pre-cached tensors for 10-50x faster data loading.

---

## Change 1: Cell 14 - Import Statement

### Find this cell:
```python
from imports.training_imports import RouterDataset
```

### Replace with:
```python
from imports.cached_dataset import CachedRouterDataset
```

---

## Change 2: Cell 18 - Remove Image Processor Loading

### Find this section:
```python
# Load image processor and tokenizer
print("Loading image processor and tokenizer...")
image_processor = CLIPImageProcessor.from_pretrained(config.vision_encoder_name)
tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer_name)
```

### Replace with:
```python
# Load tokenizer (images are pre-processed in cache)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer_name)
```

---

## Change 3: Cell 18 - Dataset Initialization

### Find this section:
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

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset:   {len(val_dataset)} samples")
print(f"Test dataset:  {len(test_dataset)} samples")
```

### Replace with:
```python
# Load cached datasets
print("\nLoading cached datasets...")

cache_dir = Path.cwd() / "cached_datasets"

# Verify cache files exist
required_caches = ["cached_train.pt", "cached_val.pt", "cached_test.pt"]
for cache_name in required_caches:
    cache_path = cache_dir / cache_name
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Please run: python 00_prefetch_and_cache_dataset.py"
        )

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

print(f"✓ Train dataset: {len(train_dataset):,} samples (cached)")
print(f"✓ Val dataset:   {len(val_dataset):,} samples (cached)")
print(f"✓ Test dataset:  {len(test_dataset):,} samples (cached)")
```

---

## Change 4 (Optional): Cell 3 or 5 - Add Import for Path

If `Path` is not already imported, add it to the imports section:

```python
from pathlib import Path
```

---

## That's It!

After making these 3-4 changes, the training notebook will use the cached datasets.

**Before running the modified notebook**, make sure you've run:
```bash
python 00_prefetch_and_cache_dataset.py
```

This creates the cache files in `cached_datasets/`.

---

## Quick Verification

After making the changes, run cells up to the dataset loading cell. You should see:

```
Loading tokenizer...
✓ Tokenizer loaded

Loading cached datasets...
Loading cached dataset from: .../cached_datasets/cached_train.pt
  Loaded 63,963 samples
  Pixel values shape: torch.Size([63963, 3, 224, 224])
Loading cached dataset from: .../cached_datasets/cached_val.pt
  Loaded 13,706 samples
  Pixel values shape: torch.Size([13706, 3, 224, 224])
Loading cached dataset from: .../cached_datasets/cached_test.pt
  Loaded 13,707 samples
  Pixel values shape: torch.Size([13707, 3, 224, 224])
✓ Train dataset: 63,963 samples (cached)
✓ Val dataset:   13,706 samples (cached)
✓ Test dataset:  13,707 samples (cached)
```

If you see this, you're ready to train with cached data! 🚀
