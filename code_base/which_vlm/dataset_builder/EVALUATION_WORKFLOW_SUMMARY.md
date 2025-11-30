# VLM Evaluation Workflow - Complete Summary

## 🎯 Your Questions Answered

### ✅ **Is the notebook optimized for all 6 models?**

**YES!** Your [fast_parallel_evaluation.ipynb](code_base/which_vlm/dataset_builder/fast_parallel_evaluation.ipynb) is **excellently optimized** with 3-level parallelism:

- **Level 1**: 10 parallel configs (ProcessPoolExecutor)
- **Level 2**: 4 parallel batches per config (ProcessPoolExecutor)
- **Level 3**: 5 parallel VLM models per batch (ThreadPoolExecutor)
- **Total**: Up to 200 concurrent inference requests

All 6 models are utilized:
- 5 VLM models run in parallel for inference
- 1 Glider model used for evaluation (separate from routing)

### ✅ **How to handle Glider timeout errors?**

**FIXED!** Multiple solutions implemented:
1. Increased timeout: 60s → 180s
2. Added control flags to disable expensive evaluations
3. Added error handling to prevent crashes
4. Fixed port mismatch (8005 → 8805)
5. Optimized vLLM server configuration

### ✅ **Can we do semantic evaluation separately after collecting results?**

**YES!** Created [semantic_evaluation_posthoc.ipynb](code_base/which_vlm/dataset_builder/semantic_evaluation_posthoc.ipynb):
- Loads already-collected VLM results
- Runs Glider evaluation on selected subsets
- No need to re-run expensive VLM inference
- Flexible sampling strategies

---

## 📋 Complete Workflow

### **Phase 1: Fast Data Collection** ⚡ (10-15 minutes)

**Notebook**: [fast_parallel_evaluation.ipynb](code_base/which_vlm/dataset_builder/fast_parallel_evaluation.ipynb)

**Configuration**:
```python
# Skip expensive evaluations
ENABLE_SEMANTIC_F1 = False
ENABLE_GLIDER_EVAL = False

# Collect all responses
N_SAMPLES_PER_CONFIG = 2000
configs_to_process = ALL_CAULDRON_CONFIGS
```

**What you get**:
- All 5 VLM model responses
- Base metrics: accuracy, exact match, contains match
- Latency and token usage statistics
- Saved to: `experiment_data/runs/exp_YYYYMMDD_HHMMSS/`

**Output files**:
```
experiment_data/runs/exp_20250127_123456/
├── docvqa.parquet
├── chartqa.parquet
├── ... (one per config)
├── all_results.parquet    ← Combined results
├── summary.json
└── COMPLETED.txt
```

---

### **Phase 2: Semantic Evaluation** 🔍 (30-60 minutes)

**Notebook**: [semantic_evaluation_posthoc.ipynb](code_base/which_vlm/dataset_builder/semantic_evaluation_posthoc.ipynb)

**Configuration**:
```python
# Input: Load Phase 1 results
USE_LATEST_RUN = True

# Enable evaluations
ENABLE_SEMANTIC_F1 = True
ENABLE_GLIDER_RUBRIC = True

# Sample strategy (choose one)
SAMPLE_STRATEGY = "per_model"
N_SAMPLES_PER_MODEL = 200
```

**What you get**:
- Semantic F1 scores (Molmo-style)
- Glider rubric scores with reasoning
- Correlation analysis
- Best/worst examples

**Output files**:
```
experiment_data/runs/exp_20250127_123456/semantic_evaluation/
├── semantic_results_semantic_20250127_153045.parquet
├── semantic_scores_semantic_20250127_153045.parquet
├── summary_semantic_20250127_153045.json
└── checkpoint_batch_XXXX.parquet (intermediate saves)
```

---

## 🚀 Action Plan

### **Step 1: Fix Glider Server** (CRITICAL)

Your Glider is on wrong port! Fix with:

```bash
# Kill old server
pkill -f "vllm serve PatronusAI/glider"

# Restart on correct port with optimizations
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

# Verify
curl http://localhost:8805/v1/models
```

### **Step 2: Run Fast Data Collection**

1. Open [fast_parallel_evaluation.ipynb](code_base/which_vlm/dataset_builder/fast_parallel_evaluation.ipynb)
2. Restart kernel (to load updated code)
3. Set configuration:
   ```python
   ENABLE_SEMANTIC_F1 = False
   ENABLE_GLIDER_EVAL = False
   N_SAMPLES_PER_CONFIG = 2000
   ```
4. Run all cells
5. Wait ~10-15 minutes
6. Results saved automatically ✅

### **Step 3: Verify Results**

```bash
cd code_base/which_vlm/dataset_builder
python verify_results.py --latest
```

This shows:
- How many configs completed
- Total records collected
- Models evaluated
- Any missing configs

### **Step 4: Run Semantic Evaluation (Optional)**

1. Open [semantic_evaluation_posthoc.ipynb](code_base/which_vlm/dataset_builder/semantic_evaluation_posthoc.ipynb)
2. Configure sampling:
   ```python
   USE_LATEST_RUN = True
   SAMPLE_STRATEGY = "per_model"
   N_SAMPLES_PER_MODEL = 200
   ```
3. Run all cells
4. Wait ~30-60 minutes
5. Analyze results ✅

---

## 📊 What Metrics You Get

### **Phase 1 Metrics** (Base Evaluation)

From `Scorer.compute_all_scores()`:

1. **Exact Match**: Response exactly equals ground truth
2. **Contains Match**: Ground truth contained in response
3. **Normalized Match**: Case-insensitive, punctuation-removed
4. **Multiple Choice Match**: Extracted letter matches
5. **Numeric Match**: Number extraction and comparison
6. **ROUGE-L**: Longest common subsequence
7. **BLEU**: N-gram precision
8. **Latency**: Inference time in ms
9. **Token Usage**: Input/output/total tokens
10. **Cost Estimation**: Based on token pricing

### **Phase 2 Metrics** (Semantic Evaluation)

From Glider evaluation:

1. **Semantic Precision**: % of generated statements that are correct
2. **Semantic Recall**: % of ground truth statements captured
3. **Semantic F1**: Harmonic mean of precision/recall
4. **Glider Score**: 0-100 rubric rating
5. **Glider Reasoning**: Explanation of score
6. **Glider Highlight**: Key issues identified

---

## 📁 File Structure

```
Which_VLM_Router/
├── code_base/which_vlm/dataset_builder/
│   ├── fast_parallel_evaluation.ipynb          ← Phase 1: Data collection
│   ├── semantic_evaluation_posthoc.ipynb       ← Phase 2: Semantic eval
│   ├── fast_parallel_evaluation_utils.py       ← Core utilities
│   ├── verify_results.py                       ← Check results script
│   ├── SEMANTIC_EVALUATION_GUIDE.md            ← Phase 2 guide
│   └── experiment_data/runs/
│       └── exp_YYYYMMDD_HHMMSS/
│           ├── *.parquet                       ← Raw results
│           └── semantic_evaluation/
│               └── *.parquet                   ← Semantic results
├── GLIDER_TIMEOUT_FIXES.md                     ← Timeout solutions
└── EVALUATION_WORKFLOW_SUMMARY.md              ← This file
```

---

## 🔧 Utilities Created

### 1. **verify_results.py**
```bash
# Check latest run
python verify_results.py --latest

# Check specific run
python verify_results.py --run-id exp_20250127_123456

# Find missing configs
python verify_results.py --latest --check-missing
```

### 2. **Resume Mode** (in notebook)
```python
# In fast_parallel_evaluation.ipynb
RESUME_MODE = True  # Skips already-completed configs
```

---

## ⚡ Performance Optimization

### Current Setup (Aggressive)
```python
MAX_WORKERS_CONFIGS = 10
MAX_WORKERS_BATCHES = 4
BATCH_SIZE = 8
# Total: 10 × 4 × 5 = 200 concurrent requests
```

### If Getting OOM Errors (Conservative)
```python
MAX_WORKERS_CONFIGS = 4
MAX_WORKERS_BATCHES = 2
BATCH_SIZE = 4
# Total: 4 × 2 × 5 = 40 concurrent requests
```

### GPU Distribution (If Multiple GPUs)
```bash
# GPU 0: Glider + small models
CUDA_VISIBLE_DEVICES=0 vllm serve PatronusAI/glider --port 8805
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8803

# GPU 1: Medium models
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8802
CUDA_VISIBLE_DEVICES=1 vllm serve deepseek-ai/DeepSeek-OCR --port 8804

# GPU 2: Large models
CUDA_VISIBLE_DEVICES=2 vllm serve google/gemma-3-27b-it --port 8800
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-VL-8B-Thinking --port 8801
```

---

## 🎓 Key Insights

### **Why Separate Evaluation?**

1. **Speed**: Get initial results in 10 minutes vs. 2+ hours
2. **Cost**: Evaluate 1000 samples instead of 100,000
3. **Flexibility**: Try different evaluation strategies
4. **Safety**: Original data preserved, can re-evaluate anytime

### **When to Use Each Notebook?**

**fast_parallel_evaluation.ipynb**:
- Collecting VLM responses
- Testing model performance
- Comparing models on base metrics
- Running full-scale evaluations

**semantic_evaluation_posthoc.ipynb**:
- Deep analysis on subsets
- Adding LLM-judge scores
- Experimenting with evaluation
- Post-hoc analysis

---

## 📈 Expected Timeline

| Phase | Task | Duration | Output |
|-------|------|----------|--------|
| 0 | Fix Glider server | 2 min | Server on port 8805 |
| 1 | Fast data collection | 10-15 min | All VLM responses |
| 2a | Analyze base metrics | Instant | Model rankings |
| 2b | Semantic eval (subset) | 30-60 min | Deep analysis |
| 3 | Final analysis | 5 min | Publication-ready results |

**Total**: ~1-2 hours for complete evaluation pipeline

---

## ✅ Success Checklist

- [ ] All 6 models running (check ports 8800-8805)
- [ ] Glider on correct port (8805, not 8005)
- [ ] Phase 1 notebook updated with control flags
- [ ] Phase 1 completed successfully
- [ ] Results verified with `verify_results.py`
- [ ] Phase 2 configured for your needs
- [ ] Phase 2 completed successfully
- [ ] Final analysis and visualizations generated

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Glider timeout | Increase REQUEST_TIMEOUT to 180s |
| Port errors | Check Glider is on 8805, not 8005 |
| OOM errors | Reduce MAX_WORKERS_CONFIGS and MAX_WORKERS_BATCHES |
| Missing results | Use `verify_results.py` to check |
| Want to resume | Set RESUME_MODE = True in notebook |
| Results disappeared | Check experiment_data/runs/ directory |

---

## 📚 Documentation Files

1. **[GLIDER_TIMEOUT_FIXES.md](GLIDER_TIMEOUT_FIXES.md)** - Timeout solutions and vLLM optimization
2. **[SEMANTIC_EVALUATION_GUIDE.md](code_base/which_vlm/dataset_builder/SEMANTIC_EVALUATION_GUIDE.md)** - Post-hoc evaluation guide
3. **[EVALUATION_WORKFLOW_SUMMARY.md](EVALUATION_WORKFLOW_SUMMARY.md)** - This file (complete overview)

---

## 🎉 Summary

You now have:

✅ **Highly optimized parallel evaluation** for all 6 models
✅ **Fixed timeout issues** with control flags and increased timeouts
✅ **Separate semantic evaluation** notebook for post-hoc analysis
✅ **Automatic result preservation** with checkpoints
✅ **Verification tools** to check progress
✅ **Resume capability** for interrupted runs
✅ **Comprehensive documentation** for all workflows

**Next Steps**: Follow the Action Plan above to collect your results! 🚀
