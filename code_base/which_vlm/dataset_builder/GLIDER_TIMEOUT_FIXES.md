# Glider Timeout Fixes & Optimization Guide

## Problem Summary
- **Issue**: Port 8805 (PatronusAI/glider) timing out during parallel evaluation
- **Root Cause**: Too many concurrent evaluation requests overwhelming the Glider model
- **Impact**: Semantic F1 and Glider rubric evaluations failing

---

## Solutions Applied

### 1. ✅ Increased Timeout
- Changed from 60s → 180s in notebook config
- Location: `fast_parallel_evaluation.ipynb` config cell

### 2. ✅ Added Evaluation Control Flags
```python
ENABLE_SEMANTIC_F1 = False  # Toggle semantic F1 computation
ENABLE_GLIDER_EVAL = False  # Toggle Glider rubric evaluation
```

**Recommendation**: Start with both **disabled**, then enable selectively after initial run.

### 3. ✅ Added Error Handling
- Wrapped Glider calls in try-catch blocks
- Evaluation failures no longer crash entire batch
- Location: `fast_parallel_evaluation_utils.py` lines 253-287

---

## Optimizing Glider vLLM Server

### Current Command (Port Mismatch Issue!)
```bash
# ❌ Your command uses port 8005, but notebook expects 8805!
CUDA_VISIBLE_DEVICES=0 nohup vllm serve PatronusAI/glider \
--trust-remote-code \
--dtype bfloat16 \
--host 0.0.0.0 \
--port 8005 \
--max-model-len 32768 \
> /home/hice1/vchopra37/scratch/logs/glider_8005.log 2>&1 &
```

### ✅ Corrected & Optimized Command
```bash
# Fix port to 8805 and optimize for evaluator workload
CUDA_VISIBLE_DEVICES=0 nohup vllm serve PatronusAI/glider \
--trust-remote-code \
--dtype bfloat16 \
--host 0.0.0.0 \
--port 8805 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--max-num-seqs 32 \
--disable-log-requests \
> /home/hice1/vchopra37/scratch/logs/glider_8805.log 2>&1 &
```

### Key Optimizations Explained:

1. **`--port 8805`** → Matches notebook configuration
2. **`--max-model-len 8192`** → Reduced from 32768 (evaluator prompts are shorter)
3. **`--max-num-seqs 32`** → Increased batch size for parallel requests (default is 256)
4. **`--gpu-memory-utilization 0.85`** → More aggressive memory usage
5. **`--disable-log-requests`** → Reduces I/O overhead

### Additional Options (if still having issues):
```bash
# For even higher throughput at cost of latency
--enable-prefix-caching \
--disable-log-stats \
--max-num-batched-tokens 8192
```

---

## Running Strategy: 3 Phases

### Phase 1: Fast Initial Run (Current Setup)
```python
ENABLE_SEMANTIC_F1 = False
ENABLE_GLIDER_EVAL = False
```
- **Goal**: Get baseline VLM performance metrics quickly
- **Metrics**: Accuracy, latency, token usage from base `Scorer`
- **Time**: ~10-15 min for all configs

### Phase 2: Glider Evaluation on Subset
```python
ENABLE_SEMANTIC_F1 = False
ENABLE_GLIDER_EVAL = True
N_SAMPLES_PER_CONFIG = 200  # Reduce sample size
```
- **Goal**: Get Glider rubric scores on smaller dataset
- **Time**: ~30-60 min

### Phase 3: Full Semantic F1 (Optional, Very Expensive)
```python
ENABLE_SEMANTIC_F1 = True
ENABLE_GLIDER_EVAL = True
N_SAMPLES_PER_CONFIG = 50  # Very small subset
```
- **Warning**: Semantic F1 makes ~10+ Glider calls per sample
- **Time**: Several hours
- **Alternative**: Run semantic F1 as post-processing on final analysis cell

---

## Ensuring Results Are Saved

### Automatic Saves (Already Implemented)
Results are saved at multiple levels:

1. **Per-Config Files**
   ```
   ./experiment_data/runs/{RUN_ID}/{config_name}.parquet
   ```

2. **Combined Results**
   ```
   ./experiment_data/runs/{RUN_ID}/all_results.parquet
   ```

3. **Summary JSON**
   ```
   ./experiment_data/runs/{RUN_ID}/summary.json
   ```

### Verify Results Exist
```bash
# Check results directory
ls -lh ./experiment_data/runs/exp_*/

# Count records in parquet files
python -c "import pandas as pd; print(pd.read_parquet('path/to/all_results.parquet').shape)"
```

### Resume from Checkpoint
If run is interrupted, you can skip completed configs:

```python
# In run cell, check existing files
completed_configs = [f.stem for f in OUTPUT_DIR.glob("*.parquet") if f.name != "all_results.parquet"]
configs_to_process = [c for c in ALL_CAULDRON_CONFIGS if c not in completed_configs]
print(f"Resuming: {len(configs_to_process)} remaining configs")
```

---

## Debugging Checklist

- [ ] Verify Glider is running on port **8805** (not 8005)
  ```bash
  curl http://localhost:8805/v1/models
  ```

- [ ] Check Glider logs for errors
  ```bash
  tail -f /home/hice1/vchopra37/scratch/logs/glider_8805.log
  ```

- [ ] Monitor GPU memory
  ```bash
  watch -n 1 nvidia-smi
  ```

- [ ] Test single inference to Glider
  ```bash
  curl http://localhost:8805/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "PatronusAI/glider",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 10
  }'
  ```

---

## Performance Tuning

### If You Have Multiple GPUs
Spread VLM models across GPUs to reduce contention:

```bash
# GPU 0: Glider (evaluator) + smaller models
CUDA_VISIBLE_DEVICES=0 vllm serve PatronusAI/glider --port 8805 &
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8803 &

# GPU 1: Medium models
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8802 &
CUDA_VISIBLE_DEVICES=1 vllm serve deepseek-ai/DeepSeek-OCR --port 8804 &

# GPU 2: Large models
CUDA_VISIBLE_DEVICES=2 vllm serve google/gemma-3-27b-it --port 8800 &
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-VL-8B-Thinking --port 8801 &
```

### Reduce Parallelism
If still getting timeouts:

```python
MAX_WORKERS_CONFIGS = 4  # Reduce from 10
MAX_WORKERS_BATCHES = 2  # Reduce from 4
BATCH_SIZE = 4           # Reduce from 8
```

---

## Summary

**Immediate Actions:**
1. ✅ Restart Glider on correct port (8805)
2. ✅ Use updated notebook with control flags
3. ✅ Start with evaluations disabled
4. ✅ Results automatically save, no extra action needed

**Next Steps:**
1. Run Phase 1 (fast baseline)
2. Review results from `all_results.parquet`
3. Enable Glider eval on subset if needed
4. Run semantic F1 as separate post-processing if required
