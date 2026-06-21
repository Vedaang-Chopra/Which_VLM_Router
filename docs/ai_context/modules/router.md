# Module: router

**Path:** `artemis_final/router/`
**Status:** PLACEHOLDER
**Owner:** routing layer
**Last scanned:** 2026-06-20

---

## Purpose

Implements **three VLM routing strategies** (Classical, Pairwise, Reward-based) with a unified interface. The router selects the optimal VLM for each (text + image) request based on accuracy/cost/latency trade-offs.

---

## Files (20)

| File | Type | Notable Findings |
|------|------|------------------|
| `__init__.py` | python | — |
| `core/__init__.py` | python | — |
| `core/api_io.py` | python | — |
| `core/config.py` | python | — |
| `core/fallback.py` | python | Placeholder return 'None' (line 223) |
| `core/inference_classical_router.py` | python | Placeholder return 'None' (line 145) |
| `core/inference_pairwise_router.py` | python | Placeholder return 'None' (line 153) |
| `core/inference_reward_router.py` | python | NOTE: Could be optimized with true batching if needed (line 370) |
| `core/legacy/inference_classical_router.py` | python | Placeholder return 'None' (line 143) |
| `core/legacy/inference_pairwise_router.py` | python | Placeholder return 'None' (line 153) |
| `core/legacy/inference_reward_router.py` | python | Placeholder return 'None' (line 144) |
| `core/router_base.py` | python | Placeholder return 'None' (line 45) |
| `core/traffic_simulator.py` | python | raise NotImplementedError (line 142) |
| `models/__init__.py` | python | — |
| `models/classical_router.py` | python | Placeholder return 'None' (line 79) |
| `models/pairwise_router.py` | python | Placeholder return 'None' (line 202) |
| `models/reward_router.py` | python | Placeholder return 'None' (line 129) |
| `notebooks/01_router_single_and_batch_modes.ipynb` | notebook | — |
| `notebooks/02_router_experiments_and_modes.ipynb` | notebook | — |
| `README.md` | markdown | — |

---

## Public API

Entry point: `artemis_final/public_api.py` → `RouterService`

```python
from artemis_final.public_api import RouterService

router = RouterService(config)
decision = router.route(sample)  # returns RouterDecision
```

---

## Routing Strategies

| Strategy | Model | Description |
|--------|-------|-------------|
| **Classical** | `classical_router.py` | Uses embeddings + classifier |
| **Pairwise** | `pairwise_router.py` | Learned pairwise preferences |
| **Reward** | `reward_router.py` | RLHF-style reward model |

All inherit from `RouterBase` (abstract base in `router_base.py`).

---

## Key Types

Defined in `core/api_io.py`:

- `RouterDecision` — chosen model, confidence, reasoning
- `Sample` — unified input (text + images + metadata)

---

## Dependencies

- `artemis_final.common.config_loader` — GlobalConfig
- `artemis_final.router_train` — trained model checkpoints
- `artemis_final.load_balancer` — for production serving

---

## Implementation Gaps (PLACEHOLDER status)

1. **7 placeholder returns** across inference files — returns `None` instead of `RouterDecision`
2. **traffic_simulator.py** — `NotImplementedError` at line 142
3. **Legacy duplicates** in `core/legacy/` — should be removed or archived
4. **Reward router batching** — TODO noted at line 370

---

## Related Modules

| Module | Relationship |
|--------|--------------|
| `router_train` | Produces model checkpoints consumed by router |
| `load_balancer` | Consumes router decisions for scheduling |
| `ares` | Evaluates router quality on benchmarks |
| `inference_engine` | Alternative inference path |
