# Post-Hoc Semantic Evaluation Guide

## Overview

The `semantic_evaluation_posthoc.ipynb` notebook allows you to run expensive Glider-based evaluations **after** you've collected VLM results, without re-running model inference.

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Fast Data Collection (fast_parallel_evaluation.ipynb)│
│   - Run with ENABLE_SEMANTIC_F1 = False                      │
│   - Run with ENABLE_GLIDER_EVAL = False                      │
│   - Collect all VLM responses (~10-15 min)                   │
│   - Save to: experiment_data/runs/exp_YYYYMMDD_HHMMSS/       │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Semantic Evaluation (semantic_evaluation_posthoc.ipynb)│
│   - Load saved results                                        │
│   - Sample subset for evaluation                             │
│   - Run Glider evaluations in controlled batches            │
│   - Add semantic scores to existing data                     │
└─────────────────────────────────────────────────────────────┘
```

## Advantages

### ✅ **Speed & Efficiency**
- **Fast initial run**: Get VLM performance metrics in 10-15 minutes
- **Selective evaluation**: Only evaluate subsets that matter
- **No re-inference**: Reuse existing VLM responses

### ✅ **Cost Savings**
- Evaluate 100 samples instead of 100,000
- Test different evaluation strategies without re-running expensive VLM inference
- Sample strategically (best/worst performers, specific models, etc.)

### ✅ **Flexibility**
- Change evaluation parameters without re-collecting data
- Run multiple evaluation experiments on same data
- Process in small batches to avoid timeouts

### ✅ **Safety**
- Original data remains untouched
- Intermediate checkpoints saved automatically
- Resume from failure without losing progress

---

## Quick Start

### 1. Collect VLM Results First

In `fast_parallel_evaluation.ipynb`:
```python
# Configuration for fast data collection
ENABLE_SEMANTIC_F1 = False  # Skip Glider evaluation
ENABLE_GLIDER_EVAL = False  # Skip Glider evaluation
N_SAMPLES_PER_CONFIG = 2000

# Run evaluation
# Results saved to: experiment_data/runs/exp_YYYYMMDD_HHMMSS/
```

### 2. Run Semantic Evaluation

Open `semantic_evaluation_posthoc.ipynb` and configure:

```python
# Input: Use latest run automatically
USE_LATEST_RUN = True

# Or specify a specific run
USE_LATEST_RUN = False
RUN_ID = "exp_20250127_123456"

# What to evaluate
ENABLE_SEMANTIC_F1 = True   # Molmo-style F1
ENABLE_GLIDER_RUBRIC = True # LLM-as-judge scoring

# How much to evaluate
SAMPLE_STRATEGY = "random"   # Options: "all", "random", "per_model", "per_config"
N_SAMPLES_TOTAL = 1000       # Sample size for random strategy
```

### 3. Run the notebook
All cells execute sequentially - it will:
- Load existing results
- Sample according to your strategy
- Run evaluation in small batches
- Save checkpoints automatically
- Generate analysis and visualizations

---

## Sampling Strategies

### Strategy 1: Random Sampling (Recommended for Quick Tests)
```python
SAMPLE_STRATEGY = "random"
N_SAMPLES_TOTAL = 1000
```
- Randomly samples 1000 records across all models and configs
- Fast overview of semantic performance
- Good for initial experiments

### Strategy 2: Per-Model Sampling (Recommended for Model Comparison)
```python
SAMPLE_STRATEGY = "per_model"
N_SAMPLES_PER_MODEL = 200
```
- Samples 200 records per model
- Ensures fair comparison across models
- Good for identifying best model

### Strategy 3: Per-Config Sampling (Recommended for Task Analysis)
```python
SAMPLE_STRATEGY = "per_config"
N_SAMPLES_PER_CONFIG = 50
```
- Samples 50 records per config (dataset)
- Analyzes performance across tasks
- Good for identifying model specialization

### Strategy 4: All Data (Use Only for Small Datasets)
```python
SAMPLE_STRATEGY = "all"
```
- Evaluates everything
- Very expensive and time-consuming
- Only use if you have < 5000 records total

---

## Advanced Filtering

### Filter by Ground Truth Type
```python
FILTER_GT_TYPES = ["freeform"]  # Only evaluate open-ended questions
# FILTER_GT_TYPES = ["exact", "numeric"]  # Only evaluate factual answers
# FILTER_GT_TYPES = None  # Evaluate all types
```

### Filter by Model
```python
FILTER_MODELS = ["gemma-3-27b", "qwen2.5-vl-7b"]  # Only these models
# FILTER_MODELS = None  # All models
```

### Filter by Config/Dataset
```python
FILTER_CONFIGS = ["docvqa", "chartqa"]  # Only these datasets
# FILTER_CONFIGS = None  # All configs
```

### Combined Example
```python
# Evaluate only freeform questions from docvqa for the best 2 models
SAMPLE_STRATEGY = "random"
N_SAMPLES_TOTAL = 500
FILTER_GT_TYPES = ["freeform"]
FILTER_CONFIGS = ["docvqa"]
FILTER_MODELS = ["gemma-3-27b", "qwen2.5-vl-7b"]
```

---

## Output Files

Results are saved to: `experiment_data/runs/{RUN_ID}/semantic_evaluation/`

### Files Created:

1. **`semantic_results_semantic_YYYYMMDD_HHMMSS.parquet`**
   - Full merged dataset with original data + semantic scores
   - Use for comprehensive analysis

2. **`semantic_scores_semantic_YYYYMMDD_HHMMSS.parquet`**
   - Evaluation scores only
   - Use for quick lookups

3. **`summary_semantic_YYYYMMDD_HHMMSS.json`**
   - Aggregated statistics
   - Mean scores by model
   - Configuration used

4. **`checkpoint_batch_XXXX.parquet`** (if SAVE_INTERMEDIATE=True)
   - Intermediate results after each batch
   - Allows resuming from failure

---

## Resuming from Failure

If evaluation is interrupted:

1. **Check completed batches:**
   ```python
   import pandas as pd
   from pathlib import Path

   checkpoint_dir = Path("experiment_data/runs/exp_20250127_123456/semantic_evaluation")
   checkpoints = sorted(checkpoint_dir.glob("checkpoint_batch_*.parquet"))

   print(f"Found {len(checkpoints)} completed batches")

   # Load all checkpoints
   dfs = [pd.read_parquet(f) for f in checkpoints]
   df_completed = pd.concat(dfs, ignore_index=True)
   print(f"Already evaluated: {len(df_completed)} records")
   ```

2. **Resume from last batch:**
   - The notebook automatically continues from where it stopped
   - Checkpoints are merged at the end

---

## Performance Tips

### Optimize Batch Size
```python
# Smaller batches = more stable, slower
BATCH_SIZE = 5

# Larger batches = faster, more memory
BATCH_SIZE = 20

# Recommended: 10-15
BATCH_SIZE = 10
```

### Adjust Timeout
```python
# If getting timeouts, increase this
REQUEST_TIMEOUT = 300  # 5 minutes

# For very long evaluations
REQUEST_TIMEOUT = 600  # 10 minutes
```

### Monitor Progress
The notebook shows:
- Records/second processing rate
- Estimated time remaining
- Batch-level progress
- Error messages for failed samples

---

## Analysis Outputs

The notebook generates:

### 1. **Semantic F1 Scores**
- Precision: How many generated statements are correct
- Recall: How many ground truth statements were captured
- F1: Harmonic mean of precision and recall

### 2. **Glider Rubric Scores**
- Score: 0-100 rating of answer quality
- Reasoning: LLM explanation of the score
- Highlight: Key issues identified

### 3. **Comparative Analysis**
- Correlation between semantic F1 and Glider scores
- Comparison with base metrics (is_correct)
- Model rankings across different metrics

### 4. **Best/Worst Examples**
- Shows top/bottom performers
- Includes Glider reasoning
- Helps identify model strengths/weaknesses

---

## Example Workflow

### Scenario: Identify Best Model for Document QA

1. **Collect data** (15 min)
   ```python
   # fast_parallel_evaluation.ipynb
   ENABLE_SEMANTIC_F1 = False
   ENABLE_GLIDER_EVAL = False
   N_SAMPLES_PER_CONFIG = 2000
   configs_to_process = ALL_CAULDRON_CONFIGS
   ```

2. **Quick baseline** (instant)
   ```python
   # Analyze base metrics from saved results
   df = pd.read_parquet("experiment_data/runs/exp_20250127_123456/all_results.parquet")
   print(df.groupby('model_name')['is_correct'].mean())
   ```

3. **Deep evaluation on top 3 models** (30 min)
   ```python
   # semantic_evaluation_posthoc.ipynb
   USE_LATEST_RUN = True
   SAMPLE_STRATEGY = "per_model"
   N_SAMPLES_PER_MODEL = 300
   FILTER_MODELS = ["gemma-3-27b", "qwen2.5-vl-7b", "qwen3-vl-8b-thinking"]
   FILTER_CONFIGS = ["docvqa", "infographic_vqa"]
   ENABLE_SEMANTIC_F1 = True
   ENABLE_GLIDER_RUBRIC = True
   ```

4. **Review results**
   - Check semantic F1 scores by model
   - Review Glider reasoning for failures
   - Identify best model for document QA

---

## Troubleshooting

### "No results found"
- Check INPUT_DIR path
- Verify RUN_ID matches folder name
- Ensure `all_results.parquet` exists

### Timeout Errors
- Increase REQUEST_TIMEOUT
- Reduce BATCH_SIZE
- Check Glider server is running on port 8805

### Memory Errors
- Reduce sample size
- Use sampling strategy instead of "all"
- Process in smaller batches

### Glider Server Not Responding
```bash
# Check Glider is running
curl http://localhost:8805/v1/models

# Restart if needed (use correct port!)
CUDA_VISIBLE_DEVICES=0 nohup vllm serve PatronusAI/glider \
--port 8805 --max-model-len 8192 \
> glider.log 2>&1 &
```

---

## Summary

✅ **Use this notebook when:**
- You have collected VLM results already
- You want to add semantic evaluation selectively
- You need to test different evaluation strategies
- You want to avoid re-running expensive VLM inference

❌ **Don't use this notebook when:**
- You haven't collected VLM results yet (use `fast_parallel_evaluation.ipynb` first)
- You want real-time evaluation during inference
- You're doing a very small initial test

**Key Benefit**: Decouple expensive VLM inference from expensive evaluation, giving you flexibility to experiment with evaluation strategies without re-collecting data.
