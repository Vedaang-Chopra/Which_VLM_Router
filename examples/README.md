# 📓 Artemis Notebooks

Centralized collection of Jupyter notebooks for the Artemis VLM Router system.

## Directory Structure

```
notebooks/
├── ares/                    # Data collection & evaluation
├── load_balancer/           # Load balancing & SLA
├── router/                  # Router inference & testing
└── router_train/            # Router training workflows
```

## Quick Start

```bash
cd notebooks
jupyter notebook
```

## Notebooks by Module

### 📊 ARES (Data & Evaluation)

| Notebook | Purpose |
|----------|---------|
| `00_pushing_*.ipynb` | Push old datasets to PostgreSQL |
| `01_parallel_inference_to_db.ipynb` | Run VLM inference at scale |
| `02_eval_scoring.ipynb` | Score responses with judges |
| `03_cost_utility_computation.ipynb` | Compute rewards & costs |
| `04_eda_evaluations.ipynb` | Explore evaluation data |
| `05_debug_failed_responses.ipynb` | Debug inference failures |

### ⚖️ Load Balancer

| Notebook | Purpose |
|----------|---------|
| `00_pipeline_tutorial.ipynb` | **Full pipeline demo** ⭐ |
| `02_load_balancer_stress_test.ipynb` | Load testing & metrics |

### 🧠 Router

| Notebook | Purpose |
|----------|---------|
| `01_understanding_router_architectures.ipynb` | Compare 3 router types |
| `02_router_unit_tests.ipynb` | Validate router functionality |
| `03_experiments_and_load_testing.ipynb` | Performance benchmarks |

### 🎓 Router Training

| Notebook | Purpose |
|----------|---------|
| `00_prepare_local_database.ipynb` | Cache data locally |
| `02_reward_router_sql_to_training.ipynb` | **Main training workflow** ⭐ |
| `03_pairwise_ranking_router.ipynb` | Train pairwise router |
| `04_classical_ce_kl_router.ipynb` | Train CE/KL router |

## Import Setup

All notebooks use this pattern for imports:

```python
import sys
from pathlib import Path

# Add artemis_final to path
sys.path.insert(0, str(Path.cwd().parent.parent))  # artemis_final/

# Now import modules
from artemis_final.router.artemis_router import RewardRouterInference
from artemis_final.load_balancer import ArtemisLoadBalancer
```

## Recommended Order

1. **Start here:** `load_balancer/00_pipeline_tutorial.ipynb` - Full demo
2. **Train router:** `router_train/02_reward_router_sql_to_training.ipynb`
3. **Test router:** `router/01_understanding_router_architectures.ipynb`
