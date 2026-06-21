# Module: data_loop
>
> Status: PLACEHOLDER
> Directory: artemis_final/data_loop/
> Entry point: collector.py
> Last updated: 2026-06-20

## Purpose

Online learning infrastructure: logs live requests to PostgreSQL, tracks routing errors, and triggers periodic router retraining.

## Entry Point

`collector.py` — DataCollector class for logging. `retrainer.py` — Retrainer class (empty).

## Public API

| Function | File | Purpose |
|---|---|---|
| `DataCollector` | `collector.py` | Log samples, responses, and routing decisions to PostgreSQL |
| `ErrorTracker` | `error_tracker.py` | Track routing errors and model failures |
| `Retrainer` | `retrainer.py` | Periodic retraining from accumulated data (INCOMPLETE) |

## Internal Structure

| File | Responsibility |
|---|---|
| `collector.py` | DataCollector — log_sample_start, log_model_response, log_routing_decision |
| `error_tracker.py` | ErrorTracker — aggregate routing errors by model and task type |
| `retrainer.py` | Retrainer — stub retrain() body (INCOMPLETE) |
| `traffic_simulator.py` | Synthetic traffic generator (NotImplementedError: see router/core/traffic_simulator.py) |

## Known Issues

- `retrainer.py` — `retrain()` body is empty. No automated retraining.
- `traffic_simulator.py` in data_loop is a stub; `router/core/traffic_simulator.py:142` has NotImplementedError.
- collector.py error path handling is incomplete.

## What an Agent Must Know

- This module depends on PostgreSQL being populated. Without evaluation data, retraining has nothing to train on.
- Retrainer needs to call `router_train/db_utils.py` to load accumulated data and then invoke training.
