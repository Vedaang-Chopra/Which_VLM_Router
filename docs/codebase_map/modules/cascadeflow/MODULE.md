# Codebase Map: cascadeflow
>
> Directory: code_base/cascadeflow/cascadeflow/cascadeflow/
> Entry point: routing/domain.py::DomainDetector
> Status: PLACEHOLDER

## Responsibility

Cascade-based routing system that queries VLMs in order of increasing cost, stopping when a model meets the quality threshold. Part of the research codebase. Provides multiple cascading strategies (quality, cost, weighted, complexity-aware) and a domain detection layer.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `routing/domain.py` | Orchestration | DomainDetector — keyword + optional ML semantic detection (INCOMPLETE) |
| `routing/cascade_pipeline.py` | Orchestration | CascadePipeline — orchestrates cascade execution |
| `routing/cascade_executor.py` | Core | CascadeExecutor — executes cascade, calls VLMs in order |
| `routing/tier_routing.py` | Core | TierRouting — tier-based routing logic |
| `routing/pre_router.py` | Core | PreRouter — pre-classification before cascade |
| `routing/tool_router.py` | Core | ToolRouter — tool-augmented routing |
| `routing/tool_complexity.py` | Utility | Complexity analysis for tool routing |
| `routing/complexity_router.py` | Core | ComplexityRouter — routing based on query complexity |
| `routing/router.py` | Runner | Main routing entry |
| `quality/quality.py` | Core | Quality-based cascade strategy (QualityCascadeStrategy) |
| `providers/base.py` | Core | Base provider class (NotImplementedError at lines 551, 611) |
| `core/` | Utility | Core utilities |
| `guardrails/` | Utility | Output guardrails |
| `integrations/` | Utility | External integrations |
| `ml/embedding.py` | Utility | Optional ML embeddings for semantic domain detection |
| `streaming/` | Utility | Streaming response handling |
| `cf_exp_2.py` | Runner | Experiment script |

## Change Guide

- **To add a new cascade strategy**: implement in `routing/` or `quality/` and add to CascadePipeline
- **To fix domain routing**: complete `domain.py::DomainDetector` and `domain.py::SemanticDomainDetector` (falls back to quality strategy)
- **To fix provider NotImplementedError**: implement stub methods in `providers/base.py`

## Call Chain

```
route(query, strategy)
  → CascadePipeline.run() (or domain-specific variant)
    → DomainDetector.detect() → QualityCascade (fallback if not implemented)
    → CascadeExecutor.execute()
      → call VLM_1 → check quality → if good enough: return
      → call VLM_2 → check quality → if good enough: return
      → ...until last VLM
```

## Dependencies

Internal: `providers`, `quality`, `ml`
External: `requests` (VLM backends), `numpy`, optional: embedding service (ml/embedding.py)
