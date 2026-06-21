# Module: Load Balancer

## What It Does

Schedules requests to VLM backends using SLA-aware capacity management. Receives the router's decision and can override it if the preferred model violates latency targets or is overloaded.

## How It Fits In

Second stage of the ARTEMIS pipeline, between Router and Inference Engine. Maintains per-model queue state and SLA monitors.

## Architecture

```mermaid
graph LR
    RO[Router Output<br/>{probs, preferred_model}] --> Sched[ArtemisLoadBalancer<br/>schedule()]
    Sched --> Check1{SLA OK?}
    Check1 -->|yes| Check2{Queue capacity<br/>available?}
    Check2 -->|yes| Select[Choose preferred]
    Check2 -->|no| Override[Override to<br/>next-best feasible]
    Check1 -->|no| Override
    Override --> Decision[SchedulingDecision<br/>{chosen_model, latency, cost}]
    Select --> Decision
    Sched --> Stats[StatsRegistry]
    Sched --> SLA[SlaMonitor]
```

## Key Files

| File | What It Does |
|---|---|
| `public_api.py` | Entry: `ArtemisLoadBalancerModule`, `init_load_balancer()`, `schedule_request()`, `simulate_traffic()` |
| `core/scheduler.py` | ArtemisLoadBalancer — capacity-aware scheduling algorithm |
| `core/types.py` | RouterOutput, SchedulingContext, SchedulingDecision, BudgetExhaustedError |
| `core/stats_registry.py` | StatsRegistry — per-(task, model) latency/cost tracking |
| `core/sla_monitor.py` | SlaMonitor — tracks SLA violations per time window |
| `core/model_state.py` | ModelStateManager — in-flight request counts per model |
| `core/config.py` | load_capacity_config, ModelCapacityConfig, GlobalSLAConfig |

## Status

**PARTIAL.** Scheduling algorithm, SLA monitoring, and StatsRegistry all work correctly. The public API (`ArtemisLoadBalancerModule`) is functional.

**Known issues:** Config override handling has TODOs (line 35, 45 of `public_api.py`). When a specific model config is passed programmatically, it may be partially ignored in favor of defaults.
