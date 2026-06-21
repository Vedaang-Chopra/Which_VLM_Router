# Module: CascadeFlow

## What It Does

A cascade-based routing system that sequentially queries VLMs in order of increasing cost, stopping when a model is "good enough" by a quality threshold. Part of the research codebase (`code_base/`).

## Key Files

| File | What It Does |
|---|---|
| `cascadeflow/routing/strategies/quality.py` | QualityCascadeStrategy — quality threshold based |
| `cascadeflow/routing/strategies/cost.py` | CostCascadeStrategy — cost-minimizing |
| `cascadeflow/routing/strategies/weighted.py` | WeightedCascadeStrategy — weighted scoring |
| `cascadeflow/routing/domain.py` | DomainCascadeStrategy (incomplete) |
| `cascadeflow/caching/response_cache.py` | Response cache (TODO: re-implement in v0.2.1) |
| `cf_exp_*.py` | Experiment scripts |

## Status

**PLACEHOLDER.** Quality, cost, and weighted cascading strategies are implemented. Domain-based routing (using classifier to predict complexity before choosing cascade) is incomplete — falls back to quality strategy. ResponseCache needs re-implementation.
