# Module: router_train
>
> Status: PLACEHOLDER
> Directory: artemis_final/router_train/
> Entry point: service.py (stub); use notebooks for actual training
> Last updated: 2026-06-20

## Purpose

Trains the router MLP from profiling data in PostgreSQL. Computes multi-objective rewards per (sample, model, mode) and trains with MSE, margin ranking, or CE/KL loss depending on router type.

## Entry Point

`service.py` — service layer (PARTIAL: has placeholder returns). **Use notebooks directly:**

- `notebooks/02_reward_router_sql_to_training.ipynb` — main training workflow
- `notebooks/05_train_multitask_reward_router.ipynb` / `v2` — multitask variants
- `notebooks/06_eval_multitask_reward_router.ipynb` — evaluation

## Public API

| Function | File | Purpose |
|---|---|---|
| `RewardRouterModel` | `models/reward_router.py` | MLP architecture definition |
| `PairwiseRouterModel` | `models/pairwise_router.py` | Margin-ranking architecture |
| `RouterModelConfig` | `config.py` | Training hyperparameters dataclass |
| `compute_rewards_*` | `reward_definitions.py` | Reward functions per mode (accuracy, cheap, fast, balanced) |
| `load_profiling_data` | `db_utils.py` | SQLAlchemy query for loading samples + responses + evals |

## Internal Structure

| File | Layer | Responsibility |
|---|---|---|
| `service.py` | Runner | Service facade (PLACEHOLDER: returns None) |
| `config.py` | Schema | RouterModelConfig dataclass |
| `db_utils.py` | Core | SQLAlchemy queries; load samples/responses/evaluations from PostgreSQL |
| `reward_definitions.py` | Core | Multi-objective reward functions; accuracy * helpfulness formulations |
| `models/reward_router.py` | Core | RewardRouterModel: DistilBERT + MLP, outputs 5 reward scores |
| `models/pairwise_router.py` | Core | PairwiseRouterModel: margin-based ranking |
| `training/dataset.py` | Core | PyTorch dataset from SQL data |
| `training/pairwise_dataset.py` | Core | Pairwise ranking dataset |
| `training/train_reward_router.py` | Runner | Training loop (used by notebooks) |
| `data/model_index.json` | Config | Model name → index mapping |
| `data/mode_index.json` | Config | Mode name → index mapping |
| `data/task_index.json` | Config | Task type → index mapping |
| `notebooks/` | Runner | Training workflows (primary entry point) |

## Dependencies

Internal: `common.config_loader`, PostgreSQL (sqlalchemy + psycopg2)
External: `torch`, `transformers` (DistilBERT), `sqlalchemy`, `numpy`

## Known Issues

- `service.py:52,102` — `return None` placeholder returns. The service layer is incomplete; do not call `service.py` methods directly.
- Use the Jupyter notebooks in `notebooks/router_train/` for training. They have the complete training loop.
- `db_utils.py:75` — `return False` placeholder for a query path.

## What an Agent Must Know Before Editing

- Reward functions in `reward_definitions.py` are the core of the training objective:
  - `accuracy` mode: `A^2 * H` (accuracy squared × helpfulness)
  - `cheap` mode: `A*H - w*(cost^e)` (accuracy × helpfulness minus cost penalty)
  - `fast` mode: `A*H - w*(latency^e)` (accuracy × helpfulness minus latency penalty)
  - `balanced` mode: multi-objective combination
- Changing reward functions requires retraining all router checkpoints.
- The MLP head dimensions must match `model_name_order` in the router config YAML.
- Checkpoints are saved to `artemis_final/checkpoints/` as `.pt` files. Hot-swap by calling `router_service.reload_model(new_path)`.
