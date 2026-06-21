# Module: frugal_gpt

**Location:** `code_base/frugal_gpt/FrugalGPT/src/FrugalGPT/`  
**Status:** PARTIAL  
**Files:** 19 (17 py, 1 md, 1 nb)

---

## Purpose

FrugalGPT — cost-optimized LLM routing from the original paper "FrugalGPT: How to Use LLMs While Reducing Cost." Implements cascade routing with cheap-to-expensive model progression.

---

## Key Files

| File | Layer | Purpose |
|------|-------|---------|
| `__init__.py` | interface | Package exports |
| `frugal_gpt.py` | core | Main FrugalGPT router class |
| `router.py` | core | Cascade routing logic |
| `models.py` | schemas | Model definitions and costs |
| `evaluate.py` | core | Evaluation pipeline |
| `cascade.py` | core | Cascade strategy implementation |
| `intro.ipynb` | notebook | Demo notebook |

---

## Entry Points

- `frugal_gpt.py` — `FrugalGPT` class with `route()`, `evaluate()`
- `router.py` — `CascadeRouter` class
- `evaluate.py` — `run_evaluation()`

---

## Key Classes / Functions

```python
class FrugalGPT:
    def __init__(self, models: List[ModelConfig], budget: float)
    def route(self, prompt: str) -> Tuple[str, float]  # returns (model, cost)
    def evaluate(self, dataset: str) -> Dict

class CascadeRouter:
    def __init__(self, stages: List[StageConfig])
    def route(self, prompt: str) -> RoutingDecision
```

---

## Data Contracts

- `models.py` — `ModelConfig(name, api_endpoint, cost_per_token, latency_ms)`
- `router.py` — `StageConfig(model, threshold, fallback)`
- Routing output → `(selected_model, estimated_cost, confidence)`

---

## Dependencies

- Internal: none
- External: openai, anthropic, requests, pandas

---

## Notable Findings (from scan)

- 19 placeholder returns in cascade logic
- 2 TODOs: "Implement adaptive thresholds", "Add latency-aware routing"
- Status: PARTIAL — core cascade works, adaptive features incomplete
