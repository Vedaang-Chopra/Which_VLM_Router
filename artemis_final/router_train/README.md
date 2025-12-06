# Artemis Router Training

Multi-objective VLM router training pipeline with reward-based optimization.

## Overview

This module implements a complete pipeline for training a transformer-based router that can intelligently select the best VLM model for a given query based on multiple reward objectives:

- **Accuracy Mode**: Maximize prediction quality
- **Cheap Mode**: Balance quality with low cost
- **Fast Mode**: Balance quality with low latency
- **Balanced Mode**: Multi-objective optimization across quality, cost, and latency

## Features

- **Multi-objective reward functions** with configurable hyperparameters
- **Transformer-based router** using pretrained text encoders (DistilBERT, etc.)
- **PostgreSQL integration** for loading profiling data
- **Comprehensive evaluation** against oracle and multiple baselines
- **Production-ready code** with type hints, logging, and error handling
- **Flexible configuration** system with CLI overrides

## Directory Structure

```
router_train/
├── __init__.py
├── config.py                    # Configuration classes
├── db_utils.py                  # Database utilities (real SQL schema)
├── reward_definitions.py        # Reward function definitions
├── requirements.txt             # Python dependencies
├── data/
│   ├── __init__.py
│   ├── router_reward_dataset.parquet    # Generated dataset
│   ├── model_index.json                  # Model name -> ID mapping
│   └── mode_index.json                   # Mode name -> ID mapping
├── models/
│   ├── __init__.py
│   ├── reward_router.py         # Router model architecture
│   └── checkpoints/
│       └── best_reward_router.pt         # Trained model
├── training/
│   ├── __init__.py
│   ├── dataset.py               # PyTorch dataset and dataloader
│   ├── train_reward_router.py   # Training loop
│   └── eval_reward_router.py    # Evaluation logic
├── notebooks/
│   ├── 02_reward_router_sql_to_training.ipynb  # Main workflow notebook ⭐
│   └── README.md                                # Notebook documentation
├── scripts/
│   ├── run_train_router.py      # Train router model (CLI)
│   ├── run_eval_router.py       # Evaluate trained router (CLI)
│   └── test_db_connection.py    # Test database connection
└── results/
    ├── eval_summary.csv                  # Evaluation results
    └── plots/                            # Visualization plots
```

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL database with profiling data
- CUDA-capable GPU (optional, but recommended)

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch`
- `transformers`
- `pandas`
- `numpy`
- `scipy`
- `sqlalchemy`
- `psycopg2-binary`
- `tqdm`
- `matplotlib`

## Database Schema

The pipeline expects 4 PostgreSQL tables:

### `vlm_sample`
- `sample_id` (TEXT): Unique sample identifier
- `prompt_raw` (TEXT): Raw prompt text
- `source_dataset` (TEXT): Source dataset name
- `router_task` (TEXT): Task category

### `vlm_responses`
- `sample_id` (FK): Reference to vlm_sample
- `model_name` (TEXT): Model identifier
- `response_text` (TEXT): Model response
- `prompt_tokens` (INT): Token count (optional)
- `response_tokens` (INT): Token count (optional)
- `cost_usd` (FLOAT): Response cost
- `latency_ms` (FLOAT): Response latency

### `vlm_evaluation`
- `sample_id` (FK): Reference to vlm_sample
- `model_name` (TEXT): Model identifier
- `exact_match` (FLOAT): Exact match score [0-1]
- `f1` (FLOAT): F1 score [0-1]
- `vqa_acc` (FLOAT): VQA accuracy [0-1]
- `critic_score` (FLOAT): Critic score
- `hallucination_score` (FLOAT): Hallucination metric

### `vlm_images`
- `sample_id` (FK): Reference to vlm_sample
- `image_path` (TEXT): Path to image (optional)
- `img_width` (INT): Image width
- `img_height` (INT): Image height

## Usage

### Recommended: Jupyter Notebook Workflow ⭐

The easiest way to get started is using the interactive notebook:

1. **Set database credentials:**
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=vlmrouter
export DB_USER=vlmrouter
export DB_PASS=your_password
```

2. **Start Jupyter:**
```bash
cd notebooks
jupyter notebook
```

3. **Open and run:** `02_reward_router_sql_to_training.ipynb`

This notebook handles the full workflow:
- Connect to PostgreSQL
- Load data from real schema (vlm_samples, vlm_responses, vlm_evaluations, vlm_images)
- Compute multi-objective rewards
- Build training dataset
- Train router model
- Evaluate and visualize results

See [notebooks/README.md](notebooks/README.md) for details.

### Alternative: Command-Line Scripts

For production/automated workflows, use the CLI scripts:

#### 1. Test Database Connection

```bash
python scripts/test_db_connection.py
```

#### 2. Train Router

```bash
python scripts/run_train_router.py \
    --epochs 10 \
    --batch-size 64 \
    --lr 3e-5 \
    --device cuda
```

### 4. Evaluate Router

Evaluate the trained router against oracle and baselines:

```bash
python scripts/run_eval_router.py
```

Options:
```bash
python scripts/run_eval_router.py \
    --model models/checkpoints/best_reward_router.pt \
    --no-plots  # Disable plot generation
```

Results are saved to:
- `results/eval_summary.csv`: Metrics table
- `results/plots/`: Visualization plots (if enabled)

## Configuration

All configuration is centralized in `config.py`. Key configuration classes:

### `DBConfig`
Database connection settings.

### `RewardWeights`
Hyperparameters for reward functions:
- `accuracy_exp`: Exponent for accuracy mode
- `cheap_cost_weight`, `cheap_cost_exp`: Cost penalty weights
- `fast_lat_weight`, `fast_lat_exp`: Latency penalty weights
- `balanced_*`: Multi-objective weights

### `RouterModelConfig`
Model architecture settings:
- `text_encoder_name`: HuggingFace model name
- `freeze_text_encoder`: Freeze encoder weights
- `model_emb_dim`, `mode_emb_dim`: Embedding dimensions
- `hidden_dim`: MLP hidden size
- `max_seq_length`: Max tokens

### `TrainingConfig`
Training hyperparameters:
- `batch_size`, `num_epochs`, `learning_rate`
- `train_ratio`, `val_ratio`, `test_ratio`: Data splits
- `device`: `"auto"`, `"cuda"`, `"cpu"`, or `"mps"`
- `early_stopping_patience`: Early stopping threshold

### `EvaluationConfig`
Evaluation settings:
- `model_size_ranking`: Model size order for baseline
- `mode_names`: Reward modes to evaluate
- `baselines`: Baseline methods
- `generate_plots`: Enable/disable plotting

## Reward Functions

The router is trained with multiple reward modes, each optimizing different objectives:

### Accuracy Mode
```python
reward = (A ** exp) * H
```
Maximizes accuracy while penalizing hallucinations.

### Cheap Mode
```python
reward = A * H - weight * (cost_norm ** exp)
```
Balances quality with low cost.

### Fast Mode
```python
reward = A * H - weight * (lat_norm ** exp)
```
Balances quality with low latency.

### Balanced Mode
```python
reward = (A ** a_exp) * H + c_weight * (C ** c_exp)
         - cost_weight * (cost ** cost_exp)
         - lat_weight * (lat ** lat_exp)
```
Multi-objective optimization across all factors.

Where:
- `A`: Primary accuracy [0-1]
- `H`: Hallucination cleanliness [0-1] (1 = clean)
- `C`: Confidence proxy [0-1]
- `cost_norm`: Normalized cost [0-1]
- `lat_norm`: Normalized latency [0-1]

## Model Architecture

The router uses a transformer-based architecture:

1. **Text Encoder**: Pretrained model (e.g., DistilBERT) encodes query + metadata
2. **Model Embedding**: Learned embeddings for each VLM model
3. **Mode Embedding**: Learned embeddings for each reward mode
4. **MLP Head**: Multi-layer perceptron predicts reward score

Input format:
```
[ROUTER] Task: {task}. Dataset: {dataset}. PromptLenWords: {len}.
ImgWidth: {w}. ImgHeight: {h}. ImgAR: {ar}. Question: {prompt}
```

## Evaluation Metrics

The evaluation compares:

### Oracle
Ground truth best model (max true reward).

### Router
Model selected by trained router (max predicted reward).

### Baselines
- **Always Biggest**: Always pick largest model
- **Always Cheapest**: Always pick cheapest model
- **Random**: Random model selection

Metrics tracked:
- Average reward
- Average accuracy
- Average cost
- Average latency
- Routing accuracy (% match with oracle)

## Extending the Pipeline

### Adding New Reward Modes

1. Define reward function in `reward_definitions.py`:
```python
def compute_reward_custom(A, H, custom_metric, weights):
    return A * H - weights.custom_weight * custom_metric
```

2. Add to `compute_rewards()` function

3. Update `build_dataset.py` with new mode ID

4. Update `config.py` evaluation mode list

### Customizing Model Architecture

Edit `models/reward_router.py`:
- Change text encoder
- Modify MLP layers
- Add attention mechanisms
- Incorporate image features

### Adding New Baselines

Edit `training/eval_reward_router.py`:
- Implement new `evaluate_baseline_*` method
- Add to `evaluate_all()` loop

## Troubleshooting

### Database Connection Issues
- Verify PostgreSQL is running
- Check credentials in environment variables
- Ensure tables exist with correct schema

### Out of Memory
- Reduce `batch_size` in config
- Use `--freeze-encoder` to freeze text encoder
- Reduce `hidden_dim` or `max_seq_length`

### Poor Router Performance
- Increase `num_epochs` for more training
- Adjust reward function weights in `config.py`
- Check for data quality issues (missing evaluations, etc.)
- Verify train/val/test splits have no leakage

### Slow Training
- Use GPU: `--device cuda`
- Increase `batch_size` if memory allows
- Reduce `num_workers` if CPU-bound
- Use smaller text encoder (e.g., `distilbert` vs `bert-base`)

## License

Part of the Artemis VLM Router project.

## Citation

If you use this code, please cite:

```bibtex
@software{artemis_router,
  title={Artemis: Multi-Objective VLM Router},
  author={Your Name},
  year={2024},
}
```
