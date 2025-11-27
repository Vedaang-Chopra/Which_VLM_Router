# Quick Start Guide - VLM Evaluation

## 🚀 5-Minute Setup

### 1. Fix Glider Server (CRITICAL!)
```bash
# Your Glider is on wrong port! Fix it now:
pkill -f "vllm serve PatronusAI/glider"

# Restart on port 8805 (not 8005!)
CUDA_VISIBLE_DEVICES=0 nohup vllm serve PatronusAI/glider \
--port 8805 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
> glider_8805.log 2>&1 &

# Verify
curl http://localhost:8805/v1/models
```

### 2. Fast Data Collection (10-15 min)
```python
# In: code_base/which_vlm/dataset_builder/fast_parallel_evaluation.ipynb

# Config cell:
ENABLE_SEMANTIC_F1 = False  # ← Skip expensive evaluation
ENABLE_GLIDER_EVAL = False  # ← Skip expensive evaluation

# Run all cells → Results saved automatically
```

### 3. Verify Results
```bash
cd code_base/which_vlm/dataset_builder
python verify_results.py --latest
```

### 4. Semantic Evaluation (Optional, 30-60 min)
```python
# In: code_base/which_vlm/dataset_builder/semantic_evaluation_posthoc.ipynb

# Config cell:
USE_LATEST_RUN = True
SAMPLE_STRATEGY = "per_model"
N_SAMPLES_PER_MODEL = 200

# Run all cells → Deep analysis results
```

---

## 📊 What You Get

### Phase 1: Fast Collection
✅ All 5 VLM model responses
✅ Base accuracy metrics
✅ Latency & token usage
✅ Model rankings
⏱️ **Time: 10-15 minutes**

### Phase 2: Deep Analysis (Optional)
✅ Semantic F1 scores
✅ Glider rubric scores
✅ LLM reasoning
✅ Best/worst examples
⏱️ **Time: 30-60 minutes**

---

## 🔥 Key Files

| File | Purpose |
|------|---------|
| [fast_parallel_evaluation.ipynb](code_base/which_vlm/dataset_builder/fast_parallel_evaluation.ipynb) | Phase 1: Collect VLM responses |
| [semantic_evaluation_posthoc.ipynb](code_base/which_vlm/dataset_builder/semantic_evaluation_posthoc.ipynb) | Phase 2: Add semantic scores |
| [verify_results.py](code_base/which_vlm/dataset_builder/verify_results.py) | Check results |
| [EVALUATION_WORKFLOW_SUMMARY.md](EVALUATION_WORKFLOW_SUMMARY.md) | Complete guide |

---

## ⚠️ Common Issues

| Problem | Fix |
|---------|-----|
| **Timeout errors** | Glider on wrong port → Use 8805 not 8005 |
| **"No models found"** | Check all 6 servers running (ports 8800-8805) |
| **OOM errors** | Reduce MAX_WORKERS_CONFIGS from 10 to 4 |
| **Results missing** | Run `verify_results.py --latest` |

---

## 💡 Pro Tips

1. **Always run Phase 1 first** (fast collection)
2. **Phase 2 is optional** (only if you need deep analysis)
3. **Results auto-save** (check `experiment_data/runs/`)
4. **Can resume interrupted runs** (set RESUME_MODE = True)
5. **Verify with script** (`verify_results.py --latest`)

---

## 📞 Need Help?

- Full workflow: [EVALUATION_WORKFLOW_SUMMARY.md](EVALUATION_WORKFLOW_SUMMARY.md)
- Timeout fixes: [GLIDER_TIMEOUT_FIXES.md](GLIDER_TIMEOUT_FIXES.md)
- Semantic eval: [SEMANTIC_EVALUATION_GUIDE.md](code_base/which_vlm/dataset_builder/SEMANTIC_EVALUATION_GUIDE.md)

---

**Ready? Start with Step 1 above! 🚀**
