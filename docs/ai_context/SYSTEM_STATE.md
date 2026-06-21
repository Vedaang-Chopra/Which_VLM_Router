# System State — ARTEMIS
>
> Last updated: 2026-06-20
> Source: docs/meta/SCAN_MANIFEST.json (376 source files scanned, 40 docs added)
> Agent: pi-cascade

## Status Table

| Component | Module | File(s) | Status | What is Actually Implemented | What is Missing or Broken |
|-----------|--------|---------|--------|------------------------------|---------------------------|
| **Router** | `artemis_final/router/` | 20 files | PARTIAL | DistilBERT + MLP inference (all 3 architectures: Reward, Pairwise, Classical); `route_request`/`route_batch` API; checkpoint loading; public API facade | `traffic_simulator.py:142` — `raise NotImplementedError` in `TrafficSimulator.run()` (traffic sim completely unavailable); `router_service.py:70,85` — placeholder returns `False`/`None`; `lb_interface.py:98` — TODO: implement Kafka producer for LB integration |
| **Load Balancer** | `artemis_final/load_balancer/` | 14 files | PARTIAL | Capacity-aware scheduling (`ArtemisLoadBalancer`); SLA monitoring (`SlaMonitor`); `StatsRegistry` for per-task latency/cost history; `RouterOutput`/`SchedulingDecision` flow; public facade; traffic simulation (`simulate_traffic`) | `public_api.py:35` — NOTE: programmatic config overrides partially ignored (uses `default_experiment_config` base instead); `public_api.py:45` — TODO: fully respect `cfg` overrides passed as objects |
| **ARES (Evaluation)** | `artemis_final/ares/` | 52 files | PLACEHOLDER | `RouterEvalPipeline` (comprehensive orchestration with ThreadPoolExecutor); `Scorer` (ground-truth accuracy/F1); `VLMJudge` (Molmo listwise ranking with image); `GliderEvaluator` (text-only fast eval); `estimate_confidence()`; SQLAlchemy DB models and operations; `cached_dataset` | `public_api.py:79,107` — `return None` placeholder returns for error paths; Glider/Molmo load heavy models at init (note in evaluation.py:32); data collection paths are partial stubs; eval pipeline depends on working inference engine |
| **Inference Engine** | `artemis_final/inference_engine/` | 9 files | PLACEHOLDER | Client structure exists (`WhichVLMClient` with LLM/VLM sub-clients) | `runners.py` — batch methods return `False`; `inference_service.py:44,50` — `return None`; `messages.py:70,73` — `return False`. **Not functional.** |
| **Router Training** | `artemis_final/router_train/` | 19 files | PLACEHOLDER | Reward/pairwise model architectures; reward functions per mode (`reward_definitions.py`); PyTorch dataset from SQL data; `db_utils.py` for SQL loading; training notebooks (`02_reward_router_sql_to_training.ipynb`, `05_*`, `06_*`) | `service.py:52,102` — `return None` placeholder returns in service layer; `db_utils.py:75` — `return False`. **Use notebooks directly.** |
| **Data Loop** | `artemis_final/data_loop/` | 5 files | PLACEHOLDER | `DataCollector` class; `ErrorTracker` class | `retrainer.py` — `retrain()` body is empty. No automated retraining. `traffic_simulator.py` in data_loop is a stub. |
| **System API** | `artemis_final/system_api/` | 4 files | COMPLETE | FastAPI app; `pipeline.py` orchestration; Pydantic schemas; `/health`, `/v1/chat/completions` endpoints; `docker-compose.yml` full stack | Depends on router + LB + IE all being functional |
| **Common** | `artemis_final/common/` | 7 files | COMPLETE | `GlobalConfig` dataclass; `load_global_config()` from YAML; all utilities | None |
| **CascadeFlow** | `code_base/cascadeflow/` | 59 files | PLACEHOLDER | QualityCascadeStrategy; CostCascadeStrategy; WeightedCascadeStrategy; `ResponseCache` skeleton; experiment scripts | DomainCascadeStrategy not implemented — TODO: convert when available (`strategies/domain.py`); ResponseCache re-implementation planned for v0.2.1; `providers/base.py:551,611` — `raise NotImplementedError` |
| **FrugalGPT** | `code_base/frugal_gpt/` | 19 files | PARTIAL | `LLMChain` orchestration; FrugalLLM service structure; model profiling | `llmchain.py:19` — `raise NotImplementedError`; `modelservice.py:59,117,122` — `raise NotImplementedError`. **Not runnable.** |
| **LOVM** | `code_base/lovm/` | 27 files | COMPLETE | LOVM orchestration; model profiling; experiments | One minor `return 0` placeholder at line 54 |
| **ARTEMIS Core** | `artemis_core/src/artemis/` | 14 files | COMPLETE | Minimal, clean implementation: config loader + inference client + load balancer + router; ~859 lines, zero findings | None |
| **ARTEMIS Entry** | `artemis_core/` | 1 file | PARTIAL | `main.py` entry point; `reproduce_experiment.sh` | Minimal wrapper; see artemis_core/src for implementation |
| **which_vlm/artemis** | `code_base/which_vlm/artemis/` | 31 files | PLACEHOLDER | Exists structurally | All public methods return `False` (lines 16, 27, 35). Duplicate of `artemis_final/router/` — do not use. |
| **which_vlm/ares** | `code_base/which_vlm/ares/` | 12 files | PLACEHOLDER | Exists structurally | Multiple `return None` placeholders (lines 87, 94, 95). Use `artemis_final/ares/` instead. |
| **which_vlm/experiments** | `code_base/which_vlm/experiments/` | 16 files | COMPLETE | Dataset + evaluation experiments | Minor `return None` at line 24 |
| **Examples** | `examples/` | 7+ files | COMPLETE | Load balancer, router, and ops examples | None |
| **Helpers** | `code_base/helpers/` | 2 files | COMPLETE | GPU metrics monitoring; LLaVA critic | None |
| **Aurelio** | `code_base/aurelio/` | 2 files | COMPLETE | Pivot dataset utilities; train/test parquet files | None |
| **Tests** | `artemis_final/tests/` | 2 files | COMPLETE | Unit tests for components | None |
| **cascade_experiments** | `artemis_final/cascade_experiments/` | 1 file | COMPLETE | Cascade comparison experiments; CSV outputs | Minor placeholder returns |
| **Documentation** | `artemis_final/` | 4 files | PARTIAL | README.md, COMPLETE_SYSTEM_OVERVIEW.md, IMPLEMENTATION_WALKTHROUGH.md, REFACTOR_NOTES.md | COMPLETE_SYSTEM_OVERVIEW.md: TODO update notebooks (line 405); TODO create FastAPI wrapper (line 406) |

---

## Verified vs Reported Results

### Verified (confirmed via end-to-end execution in code/notebooks)

| Result | Evidence | Source |
|---|---|---|
| Router inference (all 3 architectures) | `inference_reward_router.py`, `inference_pairwise_router.py`, `inference_classical_router.py` implement forward pass end-to-end | `artemis_final/router/core/` |
| Load balancer scheduling + SLA monitoring | `scheduler.py` + `sla_monitor.py` implement full scheduling loop; `public_api.py` provides `simulate_traffic` | `artemis_final/load_balancer/core/` |
| ARES evaluation (Scorer + VLMJudge) | `router_eval_pipeline.py` orchestrates full eval; `evaluation.py` implements Scorer with ground truth; `judge_molmo.py` implements Molmo judge | `artemis_final/ares/evaluation/` |
| PostgreSQL schema and operations | SQLAlchemy models defined in `db/operations.py`; `load_per_task_model_stats` and `insert_evaluations` implemented | `artemis_final/ares/db/operations.py` |
| FastAPI endpoints | `main.py` defines routes; docker-compose integrates with PostgreSQL | `artemis_final/system_api/` |
| artemis_core/src clean implementation | Zero NotImplementedError, zero placeholder returns, zero TODOs across 14 files | `artemis_core/src/artemis/` |
| Router training notebooks | `02_reward_router_sql_to_training.ipynb` et al. contain complete training loops | `artemis_final/router_train/notebooks/` |

### Reported but Unverified (described in docs/comments but not confirmed by execution)

| Result | Where Reported | What Would Confirm It |
|---|---|---|
| End-to-end cost savings from routing | `artemis_final/README.md`, `COMPLETE_SYSTEM_OVERVIEW.md` | Run production traffic through router + LB + IE; measure actual cost vs single-model baseline |
| Retraining improvement (checkpoint hot-swap) | `data_loop/retrainer.py`, `system_api/admin/retrain` | Trigger retraining; load new checkpoint; observe routing accuracy improvement |
| CascadeFlow accuracy vs ARTEMIS MLP router | `cascadeflow/` experiment scripts | Run both on same test set; compare cost-accuracy tradeoff curves |
| FrugalGPT savings vs. oracle | `frugal_gpt/` module docs | Requires completing NotImplementedError stubs first |

---

## Disabled or Bypassed Code

| File | Lines | Issue | Impact |
|---|---|---|---|
| `artemis_final/router/core/traffic_simulator.py` | 142 | `raise NotImplementedError` in `run()` | Traffic simulation completely unavailable |
| `artemis_final/router/core/lb_interface.py` | 98 | `TODO: Implement Kafka producer` | Router cannot publish routing decisions to Kafka |
| `code_base/cascadeflow/cascadeflow/cascadeflow/providers/base.py` | 551, 611 | `raise NotImplementedError` | Some cascadeflow provider methods fail |
| `code_base/frugal_gpt/FrugalGPT/src/orchestration/llmchain.py` | 19 | `raise NotImplementedError` | LLM chain orchestration fails |
| `code_base/frugal_gpt/FrugalGPT/src/service/modelservice.py` | 59, 117, 122 | `raise NotImplementedError` | FrugalGPT service not runnable |
| `artemis_final/data_loop/retrainer.py` | — | `retrain()` body is empty | No automated retraining from accumulated data |
| `artemis_final/ares/public_api.py` | 79, 107 | `return None` (silent failure) | Error paths in ARES evaluation silently pass |
| `code_base/cascadeflow/.../strategies/domain.py` | — | DomainCascadeStrategy not implemented (TODO) | Falls back to QualityCascadeStrategy; domain-aware routing unavailable |

---

## Placeholder Returns

144 total across the codebase. Key locations by module:

| Module | Count | Notable Locations |
|--------|-------|-------------------|
| **cascadeflow** | 55 | Most routing strategy methods return `None` for unimplemented paths |
| **ares** | 38 | Public API + evaluation helper methods for error cases |
| **which_vlm/ares** | 11 | Multiple `return None` in data loading |
| **which_vlm/artemis** | 8 | All public methods return `False` (lines 16, 27, 35) |
| **router** | 6 | `setup_router.py:70,85`, `core/fallback.py:223` |
| **router_train** | 5 | `service.py:52,102`, `db_utils.py:75` |
| **inference_engine** | 5 | `messages.py:70,73`, `inference_service.py:44,50` |
| **data_loop** | 4 | Data collection error paths |
| **which_vlm/inference_api_call** | 3 | Runner methods return `False` |
| **root** | 2 | `AGENTS.MD:112` (documentation rule, not code), `root level` |
| **Other** | 7 | cascade_experiments (2), helpers (1), load_balancer (1), which_vlm/experiments (1), frugal_gpt (1), lovm (1) |

---

## Safe to Build On

| Component | Status | Why |
|-----------|--------|-----|
| `artemis_final/common/` | COMPLETE | Fully working; no external dependencies on broken modules |
| `artemis_core/src/artemis/` | COMPLETE | Clean, minimal, zero findings; reference implementation |
| `artemis_final/load_balancer/` | PARTIAL | Scheduling and SLA monitoring are functional; extend here |
| `artemis_final/router/` (inference only) | PARTIAL | All 3 architectures work for inference; build evaluation/training on top |
| `artemis_final/system_api/` | COMPLETE | Endpoints exist; depend on working components downstream |
| `artemis_final/router_train/` (via notebooks) | PARTIAL | Training notebooks are functional; use them directly, not `service.py` |
| `code_base/lovm/` | COMPLETE | Fully complete; safe for benchmarking/profiling |
| `examples/` | COMPLETE | All example scripts functional |
| `artemis_final/tests/` | COMPLETE | Unit test coverage |

---

## Do Not Build On Yet

| Component | Status | Reason |
|-----------|--------|--------|
| `artemis_final/data_loop/` | PLACEHOLDER | `retrain()` empty; `traffic_simulator` NotImplementedError; no automated retraining |
| `artemis_final/inference_engine/` | PLACEHOLDER | Stub methods return `False`; client not functional; needs complete implementation |
| `artemis_final/router_train/service.py` | PLACEHOLDER | Placeholder returns; use notebooks directly |
| `artemis_final/ares/` (data collection) | PLACEHOLDER | `return None` error paths; data reliability issues; eval pipeline works but collection is unreliable |
| `artemis_final/router/core/traffic_simulator.py` | BROKEN | `raise NotImplementedError`; use `load_balancer::simulate_traffic()` instead |
| `code_base/cascadeflow/` (domain routing) | PLACEHOLDER | DomainCascadeStrategy not implemented; cascading incomplete |
| `code_base/frugal_gpt/` | PARTIAL | NotImplementedError in critical service methods; not runnable |
| `code_base/which_vlm/artemis` | PLACEHOLDER | All methods return `False`; use `artemis_final/router/` instead |
| `code_base/which_vlm/ares` | PLACEHOLDER | `return None` placeholders; use `artemis_final/ares/` instead |
