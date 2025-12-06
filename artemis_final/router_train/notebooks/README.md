# Router Training Notebooks

Interactive Jupyter notebooks for VLM router training and analysis.

## Notebooks

This directory contains three different approaches to training VLM routers:

### 02_reward_router_sql_to_training.ipynb

**Reward-Prediction Approach for VLM Routing**

This notebook demonstrates end-to-end training of a transformer-based router that predicts scalar rewards for (query, model, mode) triples.

**What it does:**
1. Connects to PostgreSQL database
2. Loads profiling data from 4 tables (vlm_samples, vlm_responses, vlm_evaluations, vlm_images)
3. Computes multi-objective rewards (accuracy, cheap, fast, balanced)
4. Builds training dataset with (sample, model, mode) → reward mapping
5. Trains router to predict rewards
6. Evaluates router vs oracle

**Key Features:**
- Real SQL schema integration
- Multi-objective reward functions
- Step-by-step explanations with plots
- Oracle vs router comparison
- Routing accuracy analysis

**Usage:**
```bash
cd router_train/notebooks
jupyter notebook 02_reward_router_sql_to_training.ipynb
```

**Requirements:**
- PostgreSQL database with profiling data
- See `../requirements.txt` for Python dependencies
- Recommended: GPU for faster training

**Expected Runtime:**
- With LIMIT=1000: ~5-10 minutes
- Full dataset: 30-60 minutes (depends on data size)

### 03_pairwise_ranking_router.ipynb

**Pairwise Ranking Approach for VLM Routing**

This notebook trains a router using pairwise comparisons - learning to rank models by comparing pairs.

**What it does:**
1. Loads profiling data and computes rewards
2. Generates pairwise examples (model_i, model_j) where model_i > model_j
3. Trains router with Margin Ranking Loss
4. Evaluates by checking if router ranks models correctly
5. Compares routing accuracy vs oracle

**Key Features:**
- Learns relative preferences instead of absolute rewards
- Uses MarginRankingLoss (encourage score_i > score_j + margin)
- Robust to reward calibration issues
- Proven approach from learning-to-rank literature

**Usage:**
```bash
cd router_train/notebooks
jupyter notebook 03_pairwise_ranking_router.ipynb
```

**Expected Runtime:**
- With LIMIT=1000: ~5-10 minutes
- Full dataset: 30-60 minutes

### 04_classical_ce_kl_router.ipynb

**Classical Classification Approach (CE + KL)**

This notebook trains a router as a standard multi-class classifier with knowledge distillation.

**What it does:**
1. Loads profiling data and computes rewards
2. Creates hard labels (best model per sample) and soft labels (probability distribution)
3. Trains with combined loss: α * CE(hard labels) + (1-α) * KL(soft labels)
4. Evaluates routing accuracy and prediction confidence
5. Compares to oracle with calibrated probabilities

**Key Features:**
- Standard classification framework (interpretable)
- Knowledge distillation from soft labels (teacher distribution)
- Calibrated softmax probabilities for confidence
- Well-studied approach with extensive literature

**Usage:**
```bash
cd router_train/notebooks
jupyter notebook 04_classical_ce_kl_router.ipynb
```

**Expected Runtime:**
- With LIMIT=1000: ~5-10 minutes
- Full dataset: 30-60 minutes

## Comparing Approaches

All three notebooks train routers on the same data but use different training objectives:

| Approach | Loss Function | Output | Best For |
|----------|--------------|--------|----------|
| **Reward-based (02)** | MSE on scalar rewards | Reward scores per model | Multi-objective optimization |
| **Pairwise (03)** | Margin Ranking Loss | Relative scores | Ranking tasks, noisy rewards |
| **Classical (04)** | CE + KL divergence | Probability distribution | Calibrated confidence, standard ML |

**Recommendation:** Run all three notebooks on your data and compare performance on a held-out test set.

## Setup

### 1. Install Dependencies

```bash
cd ..  # Go to router_train/
pip install -r requirements.txt
```

### 2. Configure Database

Set environment variables:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=vlmrouter
export DB_USER=vlmrouter
export DB_PASS=your_password
```

Or edit `config.py` directly.

### 3. Start Jupyter

```bash
jupyter notebook
```

## Tips

### Quick Testing

Set `LIMIT = 1000` in the notebook to test with a small subset:
```python
LIMIT = 1000  # Quick test
# LIMIT = None  # Full dataset
```

### Memory Issues

If you run out of memory:
- Reduce `BATCH_SIZE` (default: 32 → try 16 or 8)
- Set `NUM_EPOCHS` lower (default: 5 → try 2)
- Use CPU instead of GPU: `DEVICE = 'cpu'`

### GPU Acceleration

To use GPU:
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Check GPU is available:
```python
import torch
print(torch.cuda.is_available())
```

## Notebook Structure

Each notebook follows this structure:

1. **Setup** - Imports and configuration
2. **Connect to DB** - Database connection
3. **Load Data** - SQL queries and data loading
4. **Compute Rewards** - Multi-objective reward calculation
5. **Build Dataset** - Dataset construction and expansion
6. **Train Model** - Neural network training
7. **Evaluate** - Oracle comparison and analysis
8. **Visualize** - Plots and diagnostics

## Output

Notebooks generate:
- `../data/router_reward_dataset.parquet` - Training dataset
- `../data/model_index.json` - Model ID mappings
- `../data/mode_index.json` - Mode ID mappings
- `../models/checkpoints/best_reward_router.pt` - Trained model
- Plots and visualizations inline

## Troubleshooting

### "Module not found" errors

Make sure you're running from the notebooks directory and the path setup cell runs correctly:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
```

### Database connection fails

1. Check PostgreSQL is running
2. Verify credentials in environment variables
3. Test connection: `from db_utils import test_connection; test_connection(db_config)`

### CUDA out of memory

Reduce batch size or use CPU:
```python
BATCH_SIZE = 16  # or 8
DEVICE = 'cpu'
```

### Training is slow

- Use GPU if available
- Reduce `NUM_EPOCHS`
- Set `LIMIT` to test with smaller dataset first

## Next Steps

After running the notebook:

1. **Tune hyperparameters** - Adjust reward weights, learning rate, etc.
2. **Train longer** - Increase `NUM_EPOCHS` for better performance
3. **Try different encoders** - BERT, RoBERTa, etc.
4. **Full evaluation** - Run `../scripts/run_eval_router.py` on test set
5. **Deploy** - Use trained model for production routing

## Related Files

- `../config.py` - Configuration classes
- `../db_utils.py` - Database utilities
- `../reward_definitions.py` - Reward functions
- `../models/reward_router.py` - Router model
- `../training/dataset.py` - Dataset and dataloader
- `../README.md` - Project documentation
