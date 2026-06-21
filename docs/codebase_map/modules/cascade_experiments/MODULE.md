# Codebase Map: cascade_experiments
>
> Directory: artemis_final/cascade_experiments/
> Entry point: experiment scripts
> Status: COMPLETE

## Responsibility

Runs comparative experiments across cascade routing strategies (quality, cost, weighted, etc.) and produces CSV outputs with cost, accuracy, and latency metrics.

## File Index

| File | Layer | Purpose |
|---|---|---|
| Experiment scripts | Runner | Load test data; run each cascade strategy; write CSV results |
| `outputs/` | Data | `comparison_results_*.csv` with per-strategy metrics |

## Dependencies

Internal: `cascadeflow` (cascade strategies), `ares` (data loading)
External: `pandas`, `numpy`
