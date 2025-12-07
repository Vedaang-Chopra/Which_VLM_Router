# Artemis Load Balancer

Post-router load balancing and SLA verification for the Artemis VLM routing system.

## Overview

The load balancer takes **router predictions** and makes final model selection based on:
- **SLA constraints** (latency targets)
- **Queue capacity** (replica availability)  
- **Accuracy requirements**

```
Router (probs) → Load Balancer (schedule) → Inference (execute)
```

## Quick Start

```python
from load_balancer import (
    ArtemisLoadBalancer,
    RouterOutput,
    SchedulingContext,
    StatsRegistry,
)

# Setup
stats_registry = StatsRegistry(stats_dict)  # or load_per_task_model_stats()
model_configs = {...}  # or load_capacity_config()

# Create load balancer
lb = ArtemisLoadBalancer(
    model_configs=model_configs,
    stats_registry=stats_registry,
    global_latency_sla_ms=2000.0,
    max_accuracy_drop=0.05,
    scheduling_mode="capacity_aware"
)

# Schedule a request
router_output = RouterOutput(
    sample_id="sample_123",
    task_type="ocr",
    router_probs={"model_a": 0.3, "model_b": 0.7},
    preferred_model="model_b"
)

context = SchedulingContext(
    arrival_ts_ms=1000.0,
    load_profile="medium",
    metadata={}
)

decision = lb.schedule(router_output, context)
print(f"Chosen: {decision.chosen_model}, Latency: {decision.total_latency_ms}ms")
```

## Scheduling Modes

| Mode | Description |
|------|-------------|
| `router_only` | Always use router's preferred model (baseline) |
| `capacity_aware` | Consider SLA + accuracy constraints (default) |
| `cost_minimizing` | Pick cheapest model that satisfies constraints |

## Structure

```
load_balancer/
├── __init__.py           # Package exports
├── config.py             # Configuration classes
├── capacity_config.yaml  # Per-model capacity settings
├── types.py              # Core dataclasses
├── stats_registry.py     # Per-task/model statistics
├── model_state.py        # Queue/replica management
├── scheduler.py          # ArtemisLoadBalancer
├── sla_monitor.py        # SLA metrics computation
├── metrics_logger.py     # CSV/JSONL logging
├── wandb_logger.py       # W&B integration
├── notebooks/
│   ├── 00_pipeline_tutorial.ipynb   # Full pipeline demo
│   └── 01_analysis_template.ipynb   # Experiment analysis
└── evaluation/
    ├── __init__.py
    └── run_experiment.py  # CLI experiment runner
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `00_pipeline_tutorial.ipynb` | Router → LB → Inference walkthrough |
| `01_analysis_template.ipynb` | Load/analyze experiment results |

## Running Experiments

```bash
# Command line
python -m load_balancer.evaluation.run_experiment \
    --name my_experiment \
    --mode capacity_aware \
    --sla-ms 2000

# Results saved to: load_balancer/outputs/my_experiment/
```

## Configuration

Edit `capacity_config.yaml` to set per-model parameters:

```yaml
models:
  your_model_name:
    base_latency_ms: 500
    min_replicas: 1
    max_replicas: 5
    sla_ms: 2000
    max_qps_per_replica: 2.0
    autoscale:
      enable: true
      scale_up_latency_factor: 0.8
      scale_down_util_threshold: 0.3
      cooldown_ms: 60000
```

## Integration

The load balancer expects router output in this format:

```python
RouterOutput(
    sample_id="sample_123",
    task_type="ocr",
    router_probs={"model_a": 0.3, "model_b": 0.7},
    preferred_model="model_b"
)
```

Connect with inference using `WhichVLMClient` from `ares/inference_api_call/`.
