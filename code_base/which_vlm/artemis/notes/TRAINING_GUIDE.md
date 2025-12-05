# VLM Router Training Guide

## Overview

This guide explains how to train the VLM router using the prepared datasets with comprehensive W&B tracking.

## Files Created

1. **`train_router.py`** - Complete training script with W&B integration
2. **`06_training_router.ipynb`** - Interactive notebook version (can be generated)
3. **This guide** - Usage instructions

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision transformers wandb pillow pandas numpy matplotlib seaborn scikit-learn tqdm
```

### 2. Login to W&B (Optional but Recommended)

```bash
wandb login
```

### 3. Run Training

**Basic usage:**
```bash
cd code_base/which_vlm/artemis
python train_router.py
```

**Custom hyperparameters:**
```bash
python train_router.py --batch-size 32 --epochs 15 --lr 1e-4 --hidden-dim 384
```

**Without W&B (local only):**
```bash
python train_router.py --no-wandb
```

## Configuration

### Default Settings (Optimized from EDA)

- **Batch size**: 32
- **Learning rate**: 1e-4
- **Epochs**: 15
- **Hidden dimension**: 384
- **Fusion layers**: 4
- **Attention heads**: 8
- **Warmup steps**: 500
- **Gradient clipping**: 1.0
- **Soft label weight**: 0.3 (balances CE + KL losses)

### Model Architecture

```
Input:
  - Image → CLIP Vision Encoder (frozen) → 768D
  - Text (prompt + metadata) → DistilBERT → 768D

Processing:
  - Vision projection: 768D → 384D
  - Text projection: 768D → 384D
  - Concatenate [vision_token, text_tokens]
  - Transformer fusion: 4 layers, 8 heads
  - Classification head: 384D → 5 models

Training:
  - Loss: (1-α)*CrossEntropy + α*KL_divergence
  - Optimizer: AdamW
  - Scheduler: Linear warmup + decay
```

## What Gets Tracked (W&B)

### Training Metrics (every 50 steps)
- Loss (total, CE, KL)
- Accuracy
- Learning rate
- Gradient norms

### Validation Metrics (every epoch)
- Loss and accuracy
- Top-3 accuracy
- Per-model accuracy
- Confidence and entropy

### Visualizations
- Training curves
- Confusion matrices
- Routing distribution
- Per-task performance

### Model Artifacts
- Best model checkpoint
- Periodic checkpoints
- Final test results

## Expected Results

Based on EDA analysis with hierarchical performance + linear utility (λ=10000):

| Metric | Expected Value |
|--------|----------------|
| Training Accuracy | ~75-80% |
| Validation Accuracy | ~68-72% |
| Test Accuracy | ~68-72% |
| Mean Cost | ~$0.000037 USD/sample |
| vs Random Baseline | +48-52% |
| vs Task Heuristic | +10-15% |

### Oracle Label Distribution
```
qwen2_5_vl_3b:  43.5%  (best cost-performance)
deepseek_ocr:   28.0%  (cheap OCR tasks)
gemma_3_27b:    26.1%  (complex reasoning)
qwen2_5_vl_7b:   2.4%  (edge cases)
```

## Output Files

### Checkpoints (`./checkpoints/`)
- `best_model.pt` - Best model by validation accuracy
- `checkpoint_epoch_N.pt` - Periodic snapshots

### Logs (`./logs/`)
- `training_results.json` - Complete training summary
- Training metrics, test results, baseline comparisons

### W&B Dashboard
- Real-time training monitoring
- Interactive plots and tables
- Model comparison across runs

## Common Issues and Solutions

### 1. Out of Memory (OOM)

**Solution:** Reduce batch size
```bash
python train_router.py --batch-size 16
```

Or enable gradient accumulation (edit `RouterConfig.accumulation_steps = 2`)

### 2. Missing Images

The dataset uses placeholder black images when images can't be loaded. Check logs:
```
Dataset statistics:
  - missing_images: X
  - fallback_count: Y
```

**Solution:** Verify `image_root` path in config points to actual image directory.

### 3. Slow Training

**Solutions:**
- Increase `num_workers` for data loading (default: 4)
- Use GPU if available (automatically detected)
- Reduce `max_text_length` (default: 256)

### 4. Poor Performance

If router accuracy < task heuristic:

**Possible causes:**
- Task leakage: Router just memorizing task→model mapping
- Insufficient training: Try more epochs
- Overfitting: Add dropout, reduce model size

**Solutions:**
- Check baseline comparison in output
- Review per-task accuracy breakdown
- Try different hidden dimensions: `--hidden-dim 256` or `512`

## Advanced Usage

### Custom Configuration

Edit `train_router.py` and modify `RouterConfig` class:

```python
config = RouterConfig(
    batch_size=64,              # Larger batches
    hidden_dim=512,             # Bigger model
    num_fusion_layers=6,        # Deeper fusion
    freeze_vision=False,        # Fine-tune vision encoder
    use_soft_labels=False,      # Use only hard labels
    warmup_steps=1000,          # Longer warmup
)
```

### Multi-GPU Training

Add to `train_router.py`:

```python
model = torch.nn.DataParallel(model)
```

### Resume from Checkpoint

```python
checkpoint = torch.load('checkpoints/best_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## Evaluation After Training

### Load Best Model

```python
from train_router import VLMRouter, RouterConfig

config = RouterConfig()
model = VLMRouter(config)

checkpoint = torch.load('checkpoints/best_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inference on New Sample

```python
import torch
from transformers import CLIPImageProcessor, DistilBertTokenizer
from PIL import Image

# Load processors
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Prepare sample
image = Image.open("path/to/image.jpg")
text = "[Task: OCR. Image: 800x600] What text is in this image?"

pixel_values = image_processor(image, return_tensors='pt').pixel_values
encoding = tokenizer(text, return_tensors='pt', max_length=256, truncation=True, padding='max_length')

# Predict
with torch.no_grad():
    logits = model(pixel_values, encoding['input_ids'], encoding['attention_mask'])
    prediction = logits.argmax(dim=-1)
    probs = torch.softmax(logits, dim=-1)

print(f"Selected model: {ID_TO_NAME[prediction.item()]}")
print(f"Confidence: {probs.max():.2%}")
```

## Next Steps

1. **Run training with default settings**
   ```bash
   python train_router.py
   ```

2. **Monitor on W&B dashboard** (link printed at start)

3. **Review test results** in `logs/training_results.json`

4. **Compare with baselines**:
   - Random: ~20%
   - Task Heuristic: ~55-60%
   - Router (target): ~68-72%

5. **Iterate**:
   - Try different `hidden_dim`: 256, 384, 512
   - Adjust `soft_label_weight`: 0.0, 0.3, 0.5
   - Experiment with learning rates: 5e-5, 1e-4, 2e-4

## Key Improvements Over Original Plan

1. **Hierarchical Performance**: Uses sample + task + global priors (not just linear scoring)
2. **Soft Labels**: KL divergence loss smooths training when utilities are close
3. **Complete Tracking**: Every metric logged to W&B
4. **Baseline Comparisons**: Automatic comparison with random, heuristic, etc.
5. **Diagnostic Tools**: Confusion matrices, per-task breakdown, confidence calibration

## Questions?

Check W&B dashboard for:
- Training curves (loss, accuracy)
- Confusion matrices (router vs oracle)
- Routing distribution (which models are selected)
- Per-task accuracy (where does router excel/fail)

The training script logs everything you need to debug and understand router behavior!
