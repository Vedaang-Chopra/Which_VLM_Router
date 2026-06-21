# lovm.py
>
> Module: lovm
> Layer: Core
> Path: code_base/lovm/LOVM/LOVM/lovm.py

## Purpose

Main benchmark class for evaluating LVM rankings. Loads ground truth from `eval_table.csv`, builds pivot tables, and evaluates predicted rankings against ground truth using accuracy, Kendall's tau, and L1 loss.

## Classes

### LOVM

Benchmark evaluation class. Loads ground truth, supports three evaluation modes.

**Constructor args:**

- `pred_target` (str, default="acc1"): Which column of eval_table.csv to use as metric
- `num_rank` (int, default=5): How many top items to evaluate
- `return_mean` (bool, default=True): Add a `mean` row to output DataFrames
- `dataset_to_remove` (list): Datasets to exclude
- `model_to_remove` (list): Models to exclude

**Key methods:**

| Method | Signature | What it does |
|---|---|---|
| `evaluate_dataset_rank` | `(pred_df) -> DataFrame` | Eval dataset rankings: acc + k_tau |
| `evaluate_model_rank` | `(pred_df) -> DataFrame` | Eval model rankings: acc + k_tau |
| `evaluate_model_pred` | `(pred_df) -> DataFrame` | Eval model scores: L1 loss |
| `get_imagenet_model_rank` | `() -> DataFrame` | Baseline: ImageNet rank for all models |

### get_acc, get_k_tau, get_l1

Module-level metric functions. Each takes `pred`, `true` (pd.Series), and `num_rank`. Return scalar.

## Key Functions

| Function | Signature | What it does |
|---|---|---|
| `get_acc` | `(pred, true, num_rank)` | % of ground truth items in predicted top-N |
| `get_k_tau` | `(pred, true, num_rank)` | Kendall's tau correlation (returns 0 if ≤2 items) |
| `get_l1` | `(pred, true, num_rank)` | Mean absolute error between predicted and true scores |
| `eval_lovm` | `(pred, gt, type?)` | Standalone eval utility (partially implemented) |
| `gen_latex` | `(list)` | Format float list as LaTeX table row |
| `main` | `()` | Run ModelGPT + LOVM evaluation pipeline |

## Imports

Internal: `LOVM.constants` (NUM_RANK, GROUND_TRUTH_CSV, MODEL_NAME_COL, DATASET_COL)
External: `pandas`, `numpy`, `scipy.stats.kendalltau`, `collections.defaultdict`, `typing`

## Known Issues

- `get_k_tau()` at line 54: returns `0` when `len(true) <= 2`. Intentional — avoids NaN for very short sequences. Not a bug but worth knowing.
- `eval_lovm()` function body is incomplete — it references undefined variables (`model_rank_pred`). Use the `LOVM` class directly instead.
