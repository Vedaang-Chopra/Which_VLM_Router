# Codebase Map: router_train
>
> Directory: artemis_final/router_train/
> Entry point: notebooks/02_reward_router_sql_to_training.ipynb
> Status: PLACEHOLDER (notebooks are functional; service layer is incomplete)

## Responsibility

Trains the router MLP from profiling data stored in PostgreSQL. Computes multi-objective rewards per (sample, model, mode) and trains with MSE, margin, or CE/KL loss.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `service.py` | Runner | Service facade (PLACEHOLDER: returns None) |
| `models/reward_router.py` | Core | `RewardRouterModel` — DistilBERT + MLP architecture |
| `models/pairwise_router.py` | Core | `PairwiseRouterModel` — margin ranking architecture |
| `config.py` | Schema | `RouterModelConfig` dataclass |
| `reward_definitions.py` | Core | Multi-objective reward functions: accuracy, cheap, fast, balanced |
| `db_utils.py` | Core | SQLAlchemy queries for loading training data from PostgreSQL |
| `training/dataset.py` | Core | PyTorch dataset: (sample, model_idx, mode_idx) → reward |
| `training/pairwise_dataset.py` | Core | Pairwise ranking dataset |
| `notebooks/` | Runner | **Primary entry point**: 02_reward_router_sql_to_training.ipynb, 05_*, 06_eval_* |

## Change Guide

- **To train a router**: use `notebooks/02_reward_router_sql_to_training.ipynb` — not `service.py`
- **To add a new reward function**: edit `reward_definitions.py`; requires retraining all checkpoints
- **To change MLP dimensions**: update `models/reward_router.py` AND `router/router_config_reward.yaml`
- **To add a new router type**: add model class, add to notebook training loop, add to `router/public_api.py::load_router_from_checkpoint()`

## Dependencies

Internal: `common`, `ares` (for evaluation data), PostgreSQL
External: `torch`, `transformers` (DistilBERT), `sqlalchemy`, `psycopg2`, `numpy`
