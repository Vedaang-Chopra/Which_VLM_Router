# which_vlm Module (Legacy Entry Point)

> Status: COMPLETE — Original experimental runner
> Entry: `code_base/which_vlm/artemis/`

---

## Overview

The original experimental pipeline that evolved into artemis_final. Contains early router implementations, data loading, and evaluation experiments.

---

## Structure

```
code_base/which_vlm/artemis/
├── 00_prefetch_and_cache_dataset.py   # Data prefetching
├── 01_dataset_explore.ipynb           # Dataset exploration
├── 02_perf_function_change.ipynb      # Performance experiments
├── 03_cost_analysis.ipynb             # Cost analysis
├── constants/                         # Dataset constants
├── dataset_domains.json               # Domain definitions
├── dataset_tasks.json                 # Task definitions
├── lovm.py                            # LOVM integration
└── models.yml                         # Model configurations
```

---

## Key Experiments

| Notebook | Focus |
|----------|-------|
| `01_dataset_explore.ipynb` | Dataset statistics, domain/task distribution |
| `02_perf_function_change.ipynb` | Router performance vs model size |
| `03_cost_analysis.ipynb` | Cost-accuracy tradeoffs |

---

## Relation to artemis_final

| Legacy | artemis_final |
|--------|---------------|
| `which_vlm/artemis/` | `ares/` (evaluation) + `router/` (inference) |
| `lovm.py` | `router/core/traffic_simulator.py` |
| `models.yml` | `common/config_loader.py` |
