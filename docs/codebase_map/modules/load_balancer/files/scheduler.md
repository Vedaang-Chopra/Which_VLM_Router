# scheduler.py
>
> Module: load_balancer
> Layer: Core
> Path: artemis_final/load_balancer/core/scheduler.py

## Purpose

Core scheduling algorithm. `ArtemisLoadBalancer` receives `RouterOutput` and `SchedulingContext`, checks SLA + queue capacity, and returns `SchedulingDecision` (possibly overriding the router's preferred model).

## Classes

### ArtemisLoadBalancer

Main scheduler class. Methods: `schedule(router_output, context) -> SchedulingDecision`, `reset()`. Maintains internal `SlaMonitor` and per-model `ModelStateManager`.

## Key Functions

| Function | Signature | What it does |
|---|---|---|
| schedule | `schedule(router_output, context) -> SchedulingDecision` | Check SLA + capacity; override if overloaded; return decision |

## Imports

Internal: `load_balancer.core.types`, `load_balancer.core.stats_registry`, `load_balancer.core.sla_monitor`, `load_balancer.core.model_state`, `common.config_loader`
External: `logging`, `dataclasses`, `time`

## Known Issues

None for the core scheduling logic. Config override handling is in the public_api layer (see public_api.py:35,45).
