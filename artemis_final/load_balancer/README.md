# Artemis Load Balancer

Post-router load balancing and SLA verification for the Artemis VLM routing system.

## Overview

The `load_balancer` module is responsible for:

1. **Post-router load balancing**: Takes Artemis router predictions and makes final model selection based on capacity, SLA, and accuracy constraints
2. **Queue management**: Simulates queues and replica states for each VLM model
3. **SLA verification**: Tracks and reports latency violations, cost metrics, and accuracy
4. **Autoscaling simulation**: Dynamically adjusts replica counts based on load
5. **Comprehensive logging**: W&B integration, CSV/JSONL exports for analysis

**Note**: This module uses **simulated inference** with statistics from the Ares dataset. It does not run actual model inference.

## Module Structure

```
load_balancer/
├── __init__.py                    # Package exports
├── config.py                      # Configuration structures and defaults
├── capacity_config.yaml           # Per-model capacity and SLA settings
├── types.py                       # Core dataclasses
├── stats_registry.py              # Per-task/model statistics loader
├── model_state.py                 # Queue and replica management
├── scheduler.py                   # Main load balancer (ArtemisLoadBalancer)
├── sla_monitor.py                 # SLA metrics computation
├── metrics_logger.py              # CSV/JSONL logging
├── wandb_logger.py                # W&B integration
├── evaluation/
│   ├── __init__.py
│   ├── run_experiment.py          # Experiment orchestration
│   └── analysis_template.ipynb    # Analysis notebook template
└── README.md                      # This file
```

## Quick Start

### 1. Basic Usage

```python
from load_balancer import (
    ArtemisLoadBalancer,
    load_capacity_config,
    load_per_task_model_stats,
    StatsRegistry,
    RouterOutput,
    SchedulingContext,
)

# Load configurations
model_configs = load_capacity_config()  # from capacity_config.yaml
stats_dict = load_per_task_model_stats()  # from ares/aggregates/
stats_registry = StatsRegistry(stats_dict)

# Create load balancer
lb = ArtemisLoadBalancer(
    model_configs=model_configs,
    stats_registry=stats_registry,
    global_latency_sla_ms=2000.0,
    max_accuracy_drop=0.05,
    scheduling_mode="capacity_aware"  # or "router_only", "cost_minimizing"
)

# Schedule a request
router_output = RouterOutput(
    sample_id="sample_123",
    task_type="ocr",
    router_probs={"small_vlm": 0.2, "medium_vlm": 0.5, "large_vlm": 0.3},
    preferred_model="medium_vlm"
)

context = SchedulingContext(
    arrival_ts_ms=1000.0,
    load_profile="medium",
    metadata={}
)

decision = lb.schedule(router_output, context)

print(f"Chosen model: {decision.chosen_model}")
print(f"Predicted latency: {decision.total_latency_ms:.1f}ms")
print(f"Estimated cost: ${decision.est_cost_usd:.6f}")
```

### 2. Running Experiments

#### From Command Line

```bash
# Run with default configuration
python -m load_balancer.evaluation.run_experiment

# Customize parameters
python -m load_balancer.evaluation.run_experiment \
    --name my_experiment \
    --mode capacity_aware \
    --sla-ms 2000 \
    --max-accuracy-drop 0.05 \
    --seed 42

# Run in simulation-only mode (no queue updates)
python -m load_balancer.evaluation.run_experiment \
    --simulation-only

# Disable W&B logging
python -m load_balancer.evaluation.run_experiment --no-wandb
```

#### Programmatically

```python
from load_balancer.evaluation import run_experiment
from load_balancer.config import default_experiment_config

# Use default config
results = run_experiment()

# Or customize
config = default_experiment_config()
config.name = "my_custom_experiment"
config.scheduling_mode = "cost_minimizing"
config.random_seed = 42

results = run_experiment(experiment_config=config)

# Access results
print(f"Total requests: {results['detailed_metrics'].overall.requests}")
print(f"SLA violation rate: {results['detailed_metrics'].overall.violation_rate:.2%}")
```

### 3. Analyzing Results

Open the Jupyter notebook:

```bash
jupyter notebook load_balancer/evaluation/analysis_template.ipynb
```

Or load results programmatically:

```python
from load_balancer.metrics_logger import load_decisions_from_csv
from load_balancer.sla_monitor import compute_detailed_metrics, print_detailed_summary
from pathlib import Path

# Load decisions
csv_path = Path("load_balancer/outputs/my_experiment/decisions.csv")
decisions = load_decisions_from_csv(csv_path)

# Compute metrics
metrics = compute_detailed_metrics(decisions, latency_sla_ms=2000.0)

# Print summary
print_detailed_summary(metrics)
```

## Configuration

### Capacity Configuration (`capacity_config.yaml`)

Defines per-model capacity and autoscaling parameters:

```yaml
models:
  small_vlm:
    base_latency_ms: 250        # Average service time
    min_replicas: 1             # Minimum replicas
    max_replicas: 10            # Maximum replicas
    sla_ms: 1500                # Per-model SLA
    max_qps_per_replica: 5.0    # Throughput per replica
    autoscale:
      enable: true
      scale_up_latency_factor: 0.8    # Scale up if latency > 0.8 * sla_ms
      scale_down_util_threshold: 0.3  # Scale down if utilization < 30%
      cooldown_ms: 60000              # 60s cooldown between scaling
```

### Experiment Configuration

```python
from load_balancer.config import ExperimentConfig, LoadProfileConfig

config = ExperimentConfig(
    name="my_experiment",
    global_latency_sla_ms=2000.0,
    max_allowed_accuracy_drop=0.05,
    scheduling_mode="capacity_aware",  # "router_only", "capacity_aware", "cost_minimizing"
    simulation_only=False,
    random_seed=42,
    log_to_wandb=True,
    log_to_csv=True,
    load_profiles={
        "low": LoadProfileConfig("low", qps=2, duration_sec=60),
        "high": LoadProfileConfig("high", qps=30, duration_sec=60),
    }
)
```

## Scheduling Modes

### 1. `router_only` (Baseline)

Always uses the router's preferred model. Ignores queue state and capacity.

**Use case**: Baseline for comparison.

### 2. `capacity_aware` (Default)

Considers SLA and accuracy constraints:
- Tries models in order of router probability
- Checks SLA constraint (latency ≤ SLA)
- Checks accuracy constraint (drop ≤ max_accuracy_drop)
- Picks first model that satisfies both
- Falls back to preferred model if none satisfy

**Use case**: Production-ready load balancing.

### 3. `cost_minimizing`

Minimizes cost while satisfying constraints:
- Evaluates all models
- Filters by SLA and accuracy
- Picks cheapest among valid candidates

**Use case**: Cost optimization experiments.

## Key Components

### ArtemisLoadBalancer

Main scheduler class. See [scheduler.py](scheduler.py).

### ModelStateManager

Manages queue state and autoscaling for all models. See [model_state.py](model_state.py).

Key methods:
- `simulate_assignment()`: Predict metrics without committing
- `commit_assignment()`: Update queue state
- `_maybe_autoscale()`: Autoscaling logic

### SlaMonitor

Computes SLA metrics from scheduling decisions. See [sla_monitor.py](sla_monitor.py).

```python
from load_balancer.sla_monitor import SlaMonitor

monitor = SlaMonitor(latency_sla_ms=2000.0)

# Update incrementally
for decision in decisions:
    monitor.update(decision, load_profile="medium")

# Get snapshot
metrics = monitor.snapshot()
print(f"Violation rate: {metrics.violation_rate:.2%}")
print(f"P95 latency: {metrics.latency_p95_ms:.1f}ms")
```

### Loggers

#### CSV Logger

```python
from load_balancer.metrics_logger import CsvMetricsLogger

with CsvMetricsLogger("output.csv") as logger:
    logger.log("experiment_name", "load_profile", decision)
```

#### W&B Logger

```python
from load_balancer.wandb_logger import WandbLogger

with WandbLogger("artemis_lb", "run_name", config={}) as logger:
    logger.log_decision(decision, "load_profile", step=0)
    logger.log_sla_metrics(metrics, "load_profile", stage="final")
```

## Statistics Source

The load balancer uses per-task/model statistics from Ares:

**Location**: `artemis_final/ares/aggregates/per_task_model_stats.json`

**Format**:
```json
{
  "ocr": {
    "small_vlm": {
      "avg_latency_ms": 200.0,
      "avg_accuracy": 0.85,
      "cost_per_token_usd": 0.000002,
      "avg_tokens": 300
    },
    ...
  },
  ...
}
```

**Generation**: See Ares notebooks under `artemis_final/ares/notebooks/` for how these statistics are computed.

## Integration Points

### With Artemis Router

The load balancer expects router output in this format:

```python
RouterOutput(
    sample_id="sample_123",
    task_type="ocr",
    router_probs={"small_vlm": 0.2, "medium_vlm": 0.5, "large_vlm": 0.3},
    preferred_model="medium_vlm"  # argmax(router_probs)
)
```

**TODO**: Update `evaluation/run_experiment.py` to import the actual router:
```python
from artemis_final.router import artemis_route
```

### With Traffic Simulator

The experiment runner expects a traffic generator:

```python
def generate_traffic(load_profile_config, dataset, random_seed):
    """Yields request dicts with arrival_ts_ms."""
    ...
```

**TODO**: Update `evaluation/run_experiment.py` to import the actual traffic simulator:
```python
from artemis_final.traffic_simulator import generate_traffic
```

### With Ares Dataset

**TODO**: Update `load_dataset_samples()` in `run_experiment.py` to load actual samples:
```python
def load_dataset_samples():
    # Load from artemis_final/ares/data/
    import pandas as pd
    df = pd.read_parquet("artemis_final/ares/data/samples.parquet")
    return df.to_dict('records')
```

## Output Files

After running an experiment, you'll find:

```
load_balancer/outputs/{experiment_name}/
├── decisions.csv         # All scheduling decisions (CSV format)
└── decisions.jsonl       # All scheduling decisions (JSONL format)
```

Additionally, metrics are logged to W&B (if enabled).

## Analysis Workflow

1. **Run experiment**:
   ```bash
   python -m load_balancer.evaluation.run_experiment --name my_exp
   ```

2. **Open analysis notebook**:
   ```bash
   jupyter notebook load_balancer/evaluation/analysis_template.ipynb
   ```

3. **Update experiment name** in the notebook and run all cells

4. **Key plots**:
   - Latency distributions and percentiles
   - Cost breakdown by model/profile
   - Model usage share
   - SLA violation rates
   - Autoscaling behavior

## Troubleshooting

### Missing Stats

If you see warnings like:
```
WARNING: Missing stats for task=ocr, model=small_vlm
```

**Solution**: Run Ares notebooks to generate `per_task_model_stats.json`:
```bash
cd artemis_final/ares/notebooks
jupyter notebook 02_cost_utility_computation.ipynb
```

### W&B Not Available

If W&B is not installed:
```bash
pip install wandb
wandb login
```

Or disable W&B:
```bash
python -m load_balancer.evaluation.run_experiment --no-wandb
```

### Import Errors

Ensure you're running from the project root:
```bash
cd /path/to/Which_VLM_Router
python -m load_balancer.evaluation.run_experiment
```

## Development

### Running Tests

```bash
# TODO: Add unit tests
pytest artemis_final/load_balancer/tests/
```

### Adding a New Scheduling Mode

1. Add mode to `scheduler.py`:
   ```python
   def _schedule_my_new_mode(self, router_output, context):
       # Your logic here
       ...
   ```

2. Update `schedule()` method to dispatch to your new mode

3. Add to valid modes list and documentation

### Extending Autoscaling

Modify `model_state.py`:
```python
def _maybe_autoscale(self, model_name, predicted_latency_ms, current_time_ms):
    # Add your custom autoscaling logic
    ...
```

## References

- **Ares module**: `artemis_final/ares/` - Dataset and statistics
- **Router module**: `artemis_final/router/` - Artemis routing model
- **Traffic simulator**: (TODO: link when available)

## Contact

For questions or issues, please contact the Artemis team or file an issue in the repository.
