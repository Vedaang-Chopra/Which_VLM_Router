# Module: Router Training

## What It Does

Trains the router MLP from profiling data stored in PostgreSQL. Computes multi-objective rewards per (sample, model, mode) tuple and trains with MSE, margin, or CE/KL loss depending on architecture.

## Key Files

| File | What It Does |
|---|---|
| `service.py` | Training service entry (PLACEHOLDER returns) |
| `models/reward_router.py` | RewardRouterModel — MLP architecture |
| `models/pairwise_router.py` | PairwiseRouterModel — margin ranking |
| `config.py` | RouterModelConfig dataclass |
| `training/dataset.py` | PyTorch dataset from SQL data |
| `training/pairwise_dataset.py` | Pairwise ranking dataset |
| `reward_definitions.py` | Reward function definitions per mode |
| `db_utils.py` | SQLAlchemy query helpers for loading data |
| `notebooks/` | Main training workflows (02_reward_router_sql_to_training.ipynb) |

## Status

**PLACEHOLDER.** The model architectures, reward definitions, and notebooks are all present and functional. The `service.py` service layer has placeholder returns. Use notebooks directly for training until the service is complete.
