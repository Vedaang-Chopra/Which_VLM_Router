# Artemis Load Balancer - Quick Start Guide

Get up and running with the Artemis load balancer in 5 minutes.

## Prerequisites

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn wandb pyyaml

# Optional: W&B login
wandb login
```

## Step 1: Verify Installation

```bash
cd /path/to/Which_VLM_Router/artemis_final

# Test import
python -c "from load_balancer import ArtemisLoadBalancer; print('✓ Load balancer installed')"
```

## Step 2: Prepare Statistics (First-time only)

The load balancer needs per-task/model statistics from Ares.

### Option A: Use Existing Stats

If `artemis_final/ares/aggregates/per_task_model_stats.json` exists, you're good to go!

### Option B: Generate Stats

If the file doesn't exist, run the Ares notebooks:

```bash
cd artemis_final/ares/notebooks
jupyter notebook 02_cost_utility_computation.ipynb
# Run all cells to generate stats
```

Expected output: `artemis_final/ares/aggregates/per_task_model_stats.json`

## Step 3: Run Your First Experiment

### Quick Test (30 seconds)

```bash
cd artemis_final

# Run a quick experiment (uses mock data)
python -m load_balancer.evaluation.run_experiment \
    --name quickstart_test \
    --mode capacity_aware \
    --sla-ms 2000 \
    --seed 42 \
    --no-wandb
```

**Output**: Results saved to `load_balancer/outputs/quickstart_test/`

### Full Experiment (2-3 minutes)

```bash
python -m load_balancer.evaluation.run_experiment \
    --name my_first_experiment \
    --mode capacity_aware \
    --sla-ms 2000 \
    --max-accuracy-drop 0.05 \
    --seed 42
```

**Output**:
- CSV logs: `load_balancer/outputs/my_first_experiment/decisions.csv`
- W&B dashboard: Check your W&B project "artemis_load_balancer"

## Step 4: Analyze Results

### Option A: Jupyter Notebook (Recommended)

```bash
cd artemis_final/load_balancer/evaluation
jupyter notebook analysis_template.ipynb
```

In the notebook:
1. Update `EXPERIMENT_NAME = "my_first_experiment"`
2. Run all cells
3. View plots and metrics

### Option B: Python Script

```python
from load_balancer.metrics_logger import load_decisions_from_csv
from load_balancer.sla_monitor import compute_detailed_metrics, print_detailed_summary
from pathlib import Path

# Load results
csv_path = Path("load_balancer/outputs/my_first_experiment/decisions.csv")
decisions = load_decisions_from_csv(csv_path)

# Compute metrics
metrics = compute_detailed_metrics(decisions, latency_sla_ms=2000.0)

# Print summary
print_detailed_summary(metrics)
```

### Option C: Quick CSV Inspection

```bash
# View first 10 rows
head -n 10 load_balancer/outputs/my_first_experiment/decisions.csv

# Count total requests
wc -l load_balancer/outputs/my_first_experiment/decisions.csv
```

## Step 5: Compare Scheduling Modes

Run experiments with different modes to see the trade-offs:

```bash
# Baseline: Router-only
python -m load_balancer.evaluation.run_experiment \
    --name baseline_router_only \
    --mode router_only \
    --seed 42

# Capacity-aware (default)
python -m load_balancer.evaluation.run_experiment \
    --name capacity_aware \
    --mode capacity_aware \
    --seed 42

# Cost-minimizing
python -m load_balancer.evaluation.run_experiment \
    --name cost_minimizing \
    --mode cost_minimizing \
    --seed 42
```

Then compare in the analysis notebook!

## Common Commands

### Run with Custom Settings

```bash
# Strict SLA (500ms)
python -m load_balancer.evaluation.run_experiment \
    --sla-ms 500 \
    --mode capacity_aware

# Relaxed accuracy constraint
python -m load_balancer.evaluation.run_experiment \
    --max-accuracy-drop 0.1

# Disable autoscaling (edit capacity_config.yaml first)
# Set autoscale.enable: false for all models
```

### Load Results Programmatically

```python
from load_balancer.evaluation import run_experiment
from load_balancer.config import default_experiment_config

# Run experiment
config = default_experiment_config()
config.name = "my_custom_experiment"
config.scheduling_mode = "cost_minimizing"

results = run_experiment(experiment_config=config)

# Access metrics
print(f"Violation rate: {results['detailed_metrics'].overall.violation_rate:.2%}")
print(f"Avg cost: ${results['detailed_metrics'].overall.avg_cost_usd:.6f}")
```

### Customize Load Profiles

```python
from load_balancer.config import ExperimentConfig, LoadProfileConfig

config = ExperimentConfig(
    name="custom_load",
    global_latency_sla_ms=2000.0,
    max_allowed_accuracy_drop=0.05,
    load_profiles={
        "low": LoadProfileConfig("low", qps=5, duration_sec=30),
        "spike": LoadProfileConfig("spike", qps=100, duration_sec=10),
    }
)

from load_balancer.evaluation import run_experiment
results = run_experiment(experiment_config=config)
```

## Expected Results

After running the quick test, you should see:

### Terminal Output
```
Starting experiment: quickstart_test
Output directory: load_balancer/outputs/quickstart_test
Loading configurations...
Loaded stats for X task types
Initializing load balancer...
Running load profile: low
  QPS: 2
  Duration: 60s
  Processed 120 requests...

EXPERIMENT COMPLETE
Overall Metrics:
  Requests:          240
  Latency p50:       XX ms
  Latency p95:       XX ms
  Violation rate:    X.X%
  Avg cost:          $X.XXXXXX
  Total cost:        $X.XXXX
```

### Output Files
```
load_balancer/outputs/quickstart_test/
├── decisions.csv     (~15-20 KB for 240 requests)
└── decisions.jsonl   (~25-30 KB for 240 requests)
```

## Troubleshooting

### Error: "Stats file not found"

**Problem**: `per_task_model_stats.json` doesn't exist

**Solution**: Run Ares notebooks to generate stats (see Step 2)

### Error: "Module 'load_balancer' not found"

**Problem**: Wrong directory or Python path

**Solution**:
```bash
cd /path/to/Which_VLM_Router/artemis_final
python -m load_balancer.evaluation.run_experiment
```

### Warning: "W&B not available"

**Problem**: W&B not installed or not logged in

**Solution**:
```bash
pip install wandb
wandb login
# OR: Use --no-wandb flag to disable
```

### Error: "Invalid scheduling_mode"

**Problem**: Typo in mode name

**Solution**: Use one of: `router_only`, `capacity_aware`, `cost_minimizing`

## Next Steps

### Beginner
1. ✅ Run quick test
2. ✅ View results in notebook
3. ✅ Try different scheduling modes
4. 📖 Read [README.md](README.md) for detailed docs

### Intermediate
1. 🔧 Customize `capacity_config.yaml` (autoscaling settings)
2. 🔧 Create custom load profiles
3. 📊 Compare multiple experiments in W&B
4. 🔌 Integrate with actual Artemis router

### Advanced
1. 🧪 Add custom scheduling mode
2. 🧪 Implement custom autoscaling logic
3. 🧪 Extend metrics and logging
4. 🧪 Deploy to production

## Quick Reference Card

### CLI Commands
```bash
# Run experiment
python -m load_balancer.evaluation.run_experiment [OPTIONS]

# Options:
#   --name NAME              Experiment name
#   --mode MODE              router_only | capacity_aware | cost_minimizing
#   --sla-ms MS              SLA in milliseconds
#   --max-accuracy-drop VAL  Max accuracy drop (0.0-1.0)
#   --seed SEED              Random seed
#   --no-wandb               Disable W&B logging
#   --simulation-only        Don't commit assignments
```

### Python API
```python
from load_balancer import ArtemisLoadBalancer
from load_balancer.evaluation import run_experiment
from load_balancer.config import default_experiment_config
from load_balancer.sla_monitor import compute_detailed_metrics
from load_balancer.metrics_logger import load_decisions_from_csv
```

### Key Files
- **Config**: `load_balancer/capacity_config.yaml`
- **Stats**: `ares/aggregates/per_task_model_stats.json`
- **Output**: `load_balancer/outputs/{experiment_name}/`
- **Analysis**: `load_balancer/evaluation/analysis_template.ipynb`

---

**Need help?** Check [README.md](README.md) or [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
