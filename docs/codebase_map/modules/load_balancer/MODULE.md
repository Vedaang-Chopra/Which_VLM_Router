# Module Map: load_balancer
>
> Directory: artemis_final/load_balancer/
> Entry point: public_api.py::ArtemisLoadBalancerModule
> Status: PARTIAL

## Responsibility

Post-router capacity-aware scheduling. Verifies SLA latency targets and per-model queue availability, overrides the router's choice if constraints are violated, and tracks all scheduling decisions in the SLA monitor.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `public_api.py` | Runner | ArtemisLoadBalancerModule facade; singleton interface; TrafficSimulationResult dataclass |
| `core/scheduler.py` | Core | ArtemisLoadBalancer — capacity-aware scheduling algorithm |
| `core/types.py` | Schema | RouterOutput, SchedulingContext, SchedulingDecision, BudgetExhaustedError |
| `core/stats_registry.py` | Core | StatsRegistry — per-(task, model) latency/cost history |
| `core/sla_monitor.py` | Core | SlaMonitor — rolling window SLA violation tracking |
| `core/model_state.py` | Core | ModelStateManager — in-flight request counts per model |
| `core/config.py` | Schema | load_capacity_config, ModelCapacityConfig, GlobalSLAConfig, CAPACITY_CONFIG_PATH |

## Public API

| Function | File | Signature | Purpose |
|---|---|---|---|
| `ArtemisLoadBalancerModule` | public_api.py | `__init__(config_path?, stats_dict?)` | Create load balancer instance |
| `init_load_balancer` | public_api.py | `init_load_balancer(config_path?, stats_dict?)` | Set global singleton |
| `schedule` | public_api.py | `schedule(sample_id, task_type, router_probs, preferred_model, metadata?) -> Dict` | Main scheduling |
| `get_sla_summary` | public_api.py | `get_sla_summary() -> Dict` | Current SLA metrics |
| `simulate_traffic` | public_api.py | `simulate_traffic(lb, arrival_rate, duration_s, task_types?, pattern?) -> TrafficSimulationResult` | Traffic simulation |

## Internal Call Graph

```
schedule(sample_id, task_type, router_probs, preferred_model)
  → RouterOutput(sample_id, task_type, router_probs, preferred_model)
  → SchedulingContext(arrival_ts_ms, load_profile, metadata)
  → ArtemisLoadBalancer.schedule(router_output, context)
    → Check SLA (SlaMonitor)
    → Check queue capacity (ModelStateManager)
    → If overloaded: override to next-best feasible model
    → Update StatsRegistry with decision
  → SchedulingDecision{chosen_model, is_overloaded, est_latency_ms, ...}
```

## Dependencies

Internal: `common.config_loader`, `load_balancer.core.types`
External: `time`, `dataclasses`, `pathlib`, `logging`, `numpy` (for simulate_traffic)

## Change Guide

- **To change scheduling policy**: edit `core/scheduler.py::ArtemisLoadBalancer.schedule()`
- **To add a new scheduling mode**: add to the `scheduling_mode` parameter in scheduler and implement the logic
- **To change SLA defaults**: edit `core/config.py::default_experiment_config()`
- **To persist stats**: use `StatsRegistry.to_dict()` and save to JSON; pass `stats_dict` to constructor
