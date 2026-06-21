# stats_registry.py
>
> Module: load_balancer
> Layer: Core
> Path: artemis_final/load_balancer/core/stats_registry.py

## Purpose

Maintains per-(task_type, model) statistics for latency and cost. Loaded from JSON on init, updated after each scheduling decision, serializable for persistence.

## Classes

### StatsRegistry

Per-task-model stats. Methods: `get_stats(task_type, model)`, `update_stats(task_type, model, latency_ms, cost_usd)`, `to_dict()`, `from_dict()`.

## Imports

Internal: None
External: `json`, `pathlib`, `logging`

## Known Issues

None.
