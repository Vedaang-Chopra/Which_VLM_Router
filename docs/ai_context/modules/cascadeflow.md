# Module: cascadeflow
>
> Status: PLACEHOLDER
> Directory: code_base/cascadeflow/
> Entry point: cascadeflow/routing/domain.py
> Last updated: 2026-06-20

## Purpose

Cascade-based routing that queries VLMs sequentially from cheapest to most expensive, stopping when a model meets the quality threshold. Part of the research codebase.

## Entry Point

`cascadeflow/routing/domain.py` — DomainCascadeStrategy (incomplete; falls back to quality).

## Public API

| Class | File | Purpose |
|---|---|---|
| `QualityCascadeStrategy` | `cascadeflow/routing/strategies/quality.py` | Quality threshold-based cascade |
| `CostCascadeStrategy` | `cascadeflow/routing/strategies/cost.py` | Cost-minimizing cascade |
| `WeightedCascadeStrategy` | `cascadeflow/routing/strategies/weighted.py` | Weighted multi-objective scoring |
| `DomainCascadeStrategy` | `cascadeflow/routing/domain.py` | Complexity-aware cascade (INCOMPLETE) |

## Internal Structure

| File | Responsibility |
|---|---|
| `cascadeflow/routing/strategies/quality.py` | QualityCascadeStrategy — query by quality threshold |
| `cascadeflow/routing/strategies/cost.py` | CostCascadeStrategy — minimize cost, skip expensive if cheap suffices |
| `cascadeflow/routing/strategies/weighted.py` | WeightedCascadeStrategy — multi-objective weighted sum |
| `cascadeflow/routing/domain.py` | DomainCascadeStrategy — domain complexity classifier (TODO) |
| `cascadeflow/caching/response_cache.py` | Response cache (TODO: re-implement v0.2.1) |
| `cf_exp_*.py` | Experiment scripts |

## Dependencies

External: `numpy`, `requests` (VLM backends)

## Known Issues

- `cascadeflow/routing/domain.py` — TODO: convert to DomainCascadeStrategy when implemented. Falls back to QualityCascadeStrategy.
- `cascadeflow/caching/response_cache.py` — TODO: re-implement in v0.2.1.
- `CascadeConfig.quality_threshold` is ignored; QualityConfig uses complexity-aware thresholds instead (line 206 in config).
- DomainCascadeStrategy uses a TODO classifier approach — routing logic is incomplete.

## What an Agent Must Know

- Cascade strategies query VLMs in order (cheapest first). Unlike ARTEMIS router, this approach trades latency for potential cost savings.
- Response caching can reduce costs significantly for repeated queries — but the cache is currently non-functional.
- This is a research variant; it does not use the MLP-based approach of the main ARTEMIS router.
