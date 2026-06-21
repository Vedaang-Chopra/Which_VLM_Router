# which_vlm/artemis — Legacy Pipeline (PLACEHOLDER)

> **Status**: PLACEHOLDER — legacy implementation, not actively maintained
> **Location**: `code_base/which_vlm/artemis/`

---

## What it was

This was the **original single-file pipeline** that preceded the modular artemis_final architecture. It contains:

- `00_prefetch_and_cache_dataset.py` — dataset fetching and local caching
- `01_dataset_explore.ipynb` — EDA on the VLM dataset
- `02_perf_function_change.ipynb` — performance analysis across model sizes
- `03_cost_analysis.ipynb` — cost modeling for different routing strategies
- `datasets.txt` — list of benchmark datasets
- `eval_table.csv` / `eval_table_imagenets.csv` — evaluation results
- `lovm.py` — LOVM (Layered Optimal Vision Model) routing logic
- `models.yml` / `benchmark_rename.yml` — model configs

---

## Current state

| File | Status |
|------|--------|
| Core pipeline | Placeholder returns (8 `return False` stubs in `lovm.py`) |
| Notebooks | Runnable for historical analysis only |
| Tests | None |

---

## Why it exists

Preserved for:

- Historical comparison against modern `artemis_final/` architecture
- Reproducibility of original paper experiments
- Reference implementations of baseline routing strategies

---

## Do not use for

- New development
- Production routing
- Training pipelines (use `artemis_final/router_train/` instead)

---

## Migration path

| Legacy | Modern Replacement |
|--------|-------------------|
| `lovm.py` routing | `artemis_final/router/core/` (3 strategies) |
| `02_perf_function_change.ipynb` | `artemis_final/ares/evaluation_pipeline.py` |
| `03_cost_analysis.ipynb` | `artemis_final/load_balancer/` + router configs |

---

## Files of note

- `lovm.py:16,27,35` — 3 `return False` placeholders in core routing logic
- `eval_table.csv` — 1,847 rows of historical evaluation data
- `models.yml` — 12 model configurations (3B/7B/27B variants)
