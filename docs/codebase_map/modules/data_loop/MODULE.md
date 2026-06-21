# Codebase Map: data_loop
>
> Directory: artemis_final/data_loop/
> Entry point: collector.py::DataCollector
> Status: PLACEHOLDER

## Responsibility

Online learning infrastructure: logs requests and responses to PostgreSQL, tracks routing errors by model and task, and triggers periodic router retraining from accumulated evaluation data.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `collector.py` | Runner | `DataCollector` — log_sample_start(), log_model_response(), log_routing_decision() to PostgreSQL |
| `error_tracker.py` | Core | `ErrorTracker` — aggregate routing errors by model and task type |
| `retrainer.py` | Orchestration | `Retrainer` — periodic retraining (INCOMPLETE: retrain() body is empty) |
| `traffic_simulator.py` | Utility | Synthetic traffic generator (stub) |

## Change Guide

- **To enable automated retraining**: implement `retrainer.py::retrain()` — load accumulated data from PostgreSQL, call router_train notebooks or service, save checkpoint, signal hot-swap
- **To add more metrics tracking**: extend `error_tracker.py` with new aggregation methods

## Dependencies

Internal: `common`, `router`, `ares` (for evaluation data)
External: `sqlalchemy`, `psycopg2` (PostgreSQL)
