# Artemis Load Balancer

The **Artemis Load Balancer** makes system-aware routing decisions by balancing model preferences against real-time constraints such as latency targets, cost budgets, and queue capacities.

## Overview

While the Router suggests *which* model fits the content of a query, the Load Balancer determines *where* to dispatch it to maintain system stability. It manages traffic scheduling, enforces SLAs, and tracks system metrics.

## Architecture

- **Public API** (`public_api.py`): The primary interface for integration.
- **Scheduler** (`core/scheduler.py`): Core logic for selecting models based on the active policy (e.g., `capacity_aware`, `cost_minimizing`).
- **Stats Registry** (`core/stats_registry.py`): Stores historical performance data (latency distributions, accuracy) used for predictions.
- **Model StateManager** (`core/model_state.py`): Tracks real-time replica counts and queue depths.

## Usage

### Initialization

Initialization loads the configuration and historical statistics.

```python
from artemis_final.load_balancer.public_api import init_load_balancer

# Initialize with default or specific config
init_load_balancer(config_path="config/capacity_config.yaml")
```

### Scheduling Requests

Use `schedule_request` to get a binding decision.

```python
from artemis_final.load_balancer.public_api import schedule_request

decision = schedule_request(
    sample_id="req_123",
    task_type="vqa",
    router_probs={"model_a": 0.8, "model_b": 0.2},
    preferred_model="model_a"
)

print(f"Assigned Model: {decision['chosen_model']}")
```

### Metrics

Retrieve system performance statistics.

```python
from artemis_final.load_balancer.public_api import get_metrics

metrics = get_metrics()
print(f"Average Latency: {metrics.get('avg_latency_ms')} ms")
```

## Configuration

Configuration is managed via `artemis.yaml` (or module-specific files) and defines:

- **Model Constraints**: limits on throughput (QPS), base latency, and cost per request.
- **SLAs**: Global latency targets (e.g., 2000ms) and accuracy thresholds.
- **Modes**:
    - `router_only`: Strictly follows the router unless capacity is exceeded.
    - `capacity_aware`: Re-routes to avoid queue backups.
    - `cost_minimizing`: Optimizes primarily for budget.

## Simulation API

The module includes tools for offline load testing.

```python
from artemis_final.load_balancer.public_api import simulate_traffic

result = simulate_traffic(
    arrival_rate=5.0, # Requests per second
    duration_s=60
)
```
