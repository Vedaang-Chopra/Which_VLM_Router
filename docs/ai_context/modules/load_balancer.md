# Module: load_balancer
>
> Status: PARTIAL
> Directory: artemis_final/load_balancer/
> Entry point: public_api.py::ArtemisLoadBalancerModule
> Last updated: 2026-06-20

## Purpose

Post-router scheduling that enforces SLA latency targets and per-model queue capacity. Receives `RouterOutput` from the router and returns a `SchedulingDecision` that may override the router's choice if constraints are violated.

## Entry Point

`public_api.py` — `ArtemisLoadBalancerModule(config_path?, stats_dict?)` creates an instance. Use `init_load_balancer()` for the singleton. Call `schedule()` to get a `SchedulingDecision`.

## Public API

| Function | Signature | Purpose |
|---|---|---|
| `ArtemisLoadBalancerModule` | `__init__(config_path?, stats_dict?)` | Init scheduler, stats, and SLA monitor |
| `init_load_balancer` | `init_load_balancer(config_path?, stats_dict?)` | Set global module instance |
| `schedule` | `schedule(sample_id, task_type, router_probs, preferred_model, metadata?) -> Dict` | Main scheduling method |
| `get_sla_summary` | `get_sla_summary() -> Dict` | Current SLA metrics snapshot |
| `reset_metrics` | `reset_metrics()` | Clear accumulated metrics |
| `run_synthetic_simulation` | `run_synthetic_simulation(num_requests?) -> List[Dict]` | Fake traffic for config testing |
| `simulate_traffic` | `simulate_traffic(lb, arrival_rate, duration_s, task_types?, pattern?) -> TrafficSimulationResult` | Poisson/uniform traffic sim |
| `schedule_request` | `schedule_request(sample_id, task_type, router_probs, preferred_model)` | Singleton convenience wrapper |
| `get_metrics` | `get_metrics() -> Dict` | Singleton convenience wrapper |

## Internal Structure

| File | Layer | Responsibility |
|---|---|---|
| `public_api.py` | Runner | ArtemisLoadBalancerModule facade; singleton interface; TrafficSimulationResult dataclass |
| `core/scheduler.py` | Core | ArtemisLoadBalancer — capacity-aware scheduling algorithm |
| `core/types.py` | Schema | RouterOutput, SchedulingContext, SchedulingDecision, BudgetExhaustedError |
| `core/stats_registry.py` | Core | StatsRegistry — per-(task, model) latency/cost history |
| `core/sla_monitor.py` | Core | SlaMonitor — tracks SLA violations per rolling window |
| `core/model_state.py` | Core | ModelStateManager — in-flight request counts per model |
| `core/config.py` | Schema | load_capacity_config, ModelCapacityConfig, GlobalSLAConfig, CAPACITY_CONFIG_PATH |

## Dependencies

Internal: `common.config_loader`, `load_balancer.core.types`
External: `time`, `dataclasses`, `pathlib`, `logging`

## Known Issues

- `public_api.py:35` — NOTE: config override handling uses default_experiment_config as base; programmatic config objects may be partially ignored.
- `public_api.py:45` — TODO: fully respect `cfg` overrides passed programmatically.
- Config loading uses hard-coded defaults when files don't exist; may silently fall back to empty model configs.

## What an Agent Must Know Before Editing

- `ArtemisLoadBalancer` maintains in-memory queue state. For multi-process deployments, this state needs to be shared (e.g., via Redis) or moved to a stateful service.
- Scheduling modes: `router_only` (passthrough), `capacity_aware` (default, checks SLA + capacity), `cost_minimizing`. Default is `capacity_aware`.
- `StatsRegistry` loads historical stats from JSON. If `stats_dict` is not provided and the default path doesn't exist, it starts with empty stats and logs a warning.
- `simulate_traffic()` in `public_api.py` uses random probabilities — useful for config testing but not for accurate performance characterization.
- BudgetExhaustedError is raised when no model can meet constraints. System API should catch this and return an error response to the user.
