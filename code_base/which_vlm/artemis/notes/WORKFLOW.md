# VLM Router Optimization Workflow

## Complete Workflow Diagram

```
                    OPTIMIZATION WORKFLOW
                    ====================

┌───────────────────────────────────────────────────────────────┐
│                   CURRENT STATE (BEFORE)                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │  06_training_router.ipynb                       │           │
│  │  ┌───────────────────────────────────────┐      │           │
│  │  │  For each epoch (20 times):          │      │           │
│  │  │    For each sample (63,963):         │      │           │
│  │  │      ┌──────────────────────────┐    │      │           │
│  │  │      │ fetch_cauldron_image()   │◄───┼──────┼───────────┼─── HuggingFace
│  │  │      │ - Stream from network    │    │      │           │    (SLOW ⏱️)
│  │  │      │ - Decode image           │    │      │           │
│  │  │      │ - Process with CLIP      │    │      │           │
│  │  │      └──────────────────────────┘    │      │           │
│  │  │      Tokenize text                   │      │           │
│  │  │      Create batch                    │      │           │
│  │  └───────────────────────────────────────┘      │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  Problem: Images fetched 20× (once per epoch) = SLOW!         │
└───────────────────────────────────────────────────────────────┘

                            ↓
                     APPLY OPTIMIZATION
                            ↓

┌───────────────────────────────────────────────────────────────┐
│                 OPTIMIZED STATE (AFTER)                        │
│                                                                 │
│  ╔═══════════════════════════════════════════════════════╗    │
│  ║ STEP 1: One-time Setup (Run Once)                     ║    │
│  ║                                                        ║    │
│  ║  00_prefetch_and_cache_dataset.py                     ║    │
│  ║  ┌─────────────────────────────────────────┐          ║    │
│  ║  │ Parallel Fetching (32 workers)          │          ║    │
│  ║  │                                          │          ║    │
│  ║  │  Thread 1: ─┐                           │          ║    │
│  ║  │  Thread 2: ─┤                           │          ║    │
│  ║  │  Thread 3: ─┤ fetch_cauldron_image() ◄─┼──────────╫─── HuggingFace
│  ║  │    ...     ─┤ - Stream                 │          ║
│  ║  │  Thread 32:─┘ - Process with CLIP      │          ║    │
│  ║  │              - Save to cache            │          ║    │
│  ║  └─────────────────────────────────────────┘          ║    │
│  ║                      ↓                                 ║    │
│  ║  ┌─────────────────────────────────────────┐          ║    │
│  ║  │ cached_datasets/                        │          ║    │
│  ║  │  ├─ cached_train.pt  (~5 GB)           │          ║    │
│  ║  │  ├─ cached_val.pt    (~1 GB)           │          ║    │
│  ║  │  └─ cached_test.pt   (~1 GB)           │          ║    │
│  ║  └─────────────────────────────────────────┘          ║    │
│  ╚═══════════════════════════════════════════════════════╝    │
│                            ↓                                   │
│  ╔═══════════════════════════════════════════════════════╗    │
│  ║ STEP 2: Modified Training (Fast! ⚡)                  ║    │
│  ║                                                        ║    │
│  ║  06_training_router.ipynb (modified)                  ║    │
│  ║  ┌─────────────────────────────────────────┐          ║    │
│  ║  │  For each epoch (20 times):             │          ║    │
│  ║  │    For each sample (63,963):            │          ║    │
│  ║  │      ┌──────────────────────────┐       │          ║    │
│  ║  │      │ pixel_values[idx]        │◄──────┼──────────╫─── Disk Cache
│  ║  │      │ - Simple tensor indexing │       │          ║    (FAST ⚡)
│  ║  │      │ - Already processed      │       │          ║
│  ║  │      └──────────────────────────┘       │          ║    │
│  ║  │      Tokenize text                      │          ║    │
│  ║  │      Create batch                       │          ║    │
│  ║  └─────────────────────────────────────────┘          ║    │
│  ╚═══════════════════════════════════════════════════════╝    │
│                                                                 │
│  Result: Images loaded from cache = 10-50× FASTER! 🚀         │
└───────────────────────────────────────────────────────────────┘
```

---

## Detailed Execution Flow

### Phase 1: Caching (One-time)

```
00_prefetch_and_cache_dataset.py
│
├─► Load datasets
│   ├─ router_train_final.parquet (63,963 samples)
│   ├─ router_val_final.parquet   (13,706 samples)
│   └─ router_test_final.parquet  (13,707 samples)
│
├─► Initialize CLIP processor
│   └─ openai/clip-vit-base-patch32
│
├─► Process Train Split
│   ├─ Create thread pool (32 workers)
│   ├─ For each sample in parallel:
│   │   ├─ fetch_cauldron_image(source_config, source_index)
│   │   │   ├─ Check local cache (image_root/config/hash.png)
│   │   │   ├─ If not cached: stream from HuggingFace
│   │   │   ├─ Save to local cache
│   │   │   └─ Return PIL.Image
│   │   ├─ Process with CLIP processor
│   │   └─ Return pixel_values tensor [3, 224, 224]
│   ├─ Stack all tensors → [N, 3, 224, 224]
│   └─ Save to cached_train.pt
│       ├─ pixel_values: Tensor
│       ├─ dataframe: pd.DataFrame
│       └─ config: Dict
│
├─► Process Val Split
│   └─ (same as train) → cached_val.pt
│
└─► Process Test Split
    └─ (same as train) → cached_test.pt
```

### Phase 2: Training (Fast!)

```
06_training_router.ipynb (modified)
│
├─► Load cached datasets
│   ├─ CachedRouterDataset(cached_train.pt)
│   │   ├─ Load pixel_values tensor (all images)
│   │   └─ Load dataframe (all metadata)
│   ├─ CachedRouterDataset(cached_val.pt)
│   └─ CachedRouterDataset(cached_test.pt)
│
├─► Create DataLoaders
│   └─ DataLoader(train_dataset, batch_size=32, ...)
│
└─► Training Loop
    ├─ For epoch in range(20):
    │   ├─ For batch in train_loader:
    │   │   ├─ Get batch samples
    │   │   │   ├─ pixel_values = self.pixel_values[idx]  ← Fast! Just indexing
    │   │   │   ├─ Tokenize text (on-the-fly, fast)
    │   │   │   └─ Return batch
    │   │   ├─ Forward pass
    │   │   ├─ Backward pass
    │   │   └─ Update weights
    │   └─ Validation
    └─ Test evaluation
```

---

## File Dependencies

```
Project Structure:
├── 00_prefetch_and_cache_dataset.py  ← Run this first
│   └── Uses:
│       ├── imports/check_data_utils.py (fetch_cauldron_image)
│       └── transformers.CLIPImageProcessor
│
├── cached_datasets/  ← Created by caching script
│   ├── cached_train.pt
│   ├── cached_val.pt
│   └── cached_test.pt
│
├── imports/
│   ├── cached_dataset.py  ← CachedRouterDataset class
│   │   └── Loads .pt files
│   └── check_data_utils.py  ← Cauldron fetching logic
│
├── 06_training_router.ipynb  ← Modified training notebook
│   └── Uses: CachedRouterDataset
│
└── verify_cache.py  ← Verification script
    └── Checks cache integrity
```

---

## Data Flow

### Original (Slow)

```
Training Step
  ↓
DataLoader.__getitem__(idx)
  ↓
RouterDataset.__getitem__(idx)
  ↓
fetch_cauldron_image()
  ↓
┌─────────────────────────┐
│ HuggingFace Streaming   │  ← NETWORK CALL (Slow!)
│ - datasets.load_dataset │
│ - itertools.islice      │
│ - stream to index       │
└─────────────────────────┘
  ↓
PIL.Image
  ↓
CLIPImageProcessor()
  ↓
pixel_values tensor
  ↓
Return to DataLoader
```

**Time per sample:** ~100-500ms (depending on network)

### Optimized (Fast)

```
Training Step
  ↓
DataLoader.__getitem__(idx)
  ↓
CachedRouterDataset.__getitem__(idx)
  ↓
self.pixel_values[idx]  ← TENSOR INDEXING (Fast!)
  ↓
pixel_values tensor
  ↓
Return to DataLoader
```

**Time per sample:** ~0.1-1ms (100-500× faster!)

---

## Cache Structure

```
cached_train.pt (torch.save)
│
├── 'pixel_values': Tensor[63963, 3, 224, 224]
│   │
│   ├─ Index 0: [3, 224, 224]  ← Pre-processed image
│   ├─ Index 1: [3, 224, 224]
│   ├─ Index 2: [3, 224, 224]
│   └─ ...
│
├── 'dataframe': pd.DataFrame (63963 rows)
│   │
│   ├─ sample_id
│   ├─ source_config
│   ├─ router_task
│   ├─ prompt_raw
│   ├─ router_best_model_id
│   ├─ router_soft_p_*
│   └─ ... (all metadata)
│
└── 'config': Dict
    ├─ vision_encoder_name: "openai/clip-vit-base-patch32"
    ├─ n_samples: 63963
    └─ n_errors: 0
```

---

## Time Breakdown

### One-time Setup

```
00_prefetch_and_cache_dataset.py
│
├─ Load datasets:              ~5 seconds
├─ Initialize CLIP processor:  ~2 seconds
├─ Fetch & process images:     ~15-30 minutes
│   ├─ Train (63,963):  ~10-20 min
│   ├─ Val   (13,706):  ~3-5 min
│   └─ Test  (13,707):  ~3-5 min
└─ Save to disk:               ~10 seconds
                               ──────────
Total:                         ~15-30 minutes (one time!)
```

### Training (Per Epoch)

**Before (original):**
```
├─ Load images:  ~5-10 minutes  ← SLOW
└─ Training:     ~3-5 minutes
   ──────────
   Total:       ~8-15 minutes per epoch
```

**After (optimized):**
```
├─ Load images:  ~10-30 seconds  ← FAST!
└─ Training:     ~3-5 minutes
   ──────────
   Total:       ~3-6 minutes per epoch
```

**Savings:** ~5-10 minutes per epoch × 20 epochs = **~2-3 hours saved!**

---

## Summary

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Image Loading** | Stream from network | Load from disk | 10-50× faster |
| **Per Epoch Time** | 8-15 min | 3-6 min | 2-3× faster |
| **Total Training (20 epochs)** | 3-5 hours | 1-2 hours | 2-3× faster |
| **Setup Required** | None | 15-30 min (once) | One-time cost |
| **Disk Space** | Minimal | ~6-7 GB | Trade-off |

---

## Next Steps

1. **Run caching script:**
   ```bash
   python 00_prefetch_and_cache_dataset.py
   ```

2. **Verify:**
   ```bash
   python verify_cache.py
   ```

3. **Modify training notebook** (see `TRAINING_NOTEBOOK_CHANGES.md`)

4. **Train faster!** 🚀
