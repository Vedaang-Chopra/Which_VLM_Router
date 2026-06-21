# AI Context Index — ARTEMIS
>
> Last updated: 2026-06-20
> Full scan: docs/meta/SCAN_MANIFEST.json (416 files: 376 source + 40 docs)
> System state: docs/ai_context/SYSTEM_STATE.md
> 42 modules | 29 COMPLETE, 3 PARTIAL, 10 PLACEHOLDER

## What This System Does

ARTEMIS is a cost-aware Vision-Language Model (VLM) router that uses frozen CLIP + DistilBERT encoders and an MLP classifier to dispatch multimodal queries to the cheapest model meeting accuracy constraints, routing across five VLMs: deepseek_ocr, qwen2_5_vl_3b, qwen2_5_vl_7b, qwen3_vl_8b_thinking, and gemma_3_27b. It supports four routing modes (accuracy, cheap, fast, balanced) and uses a PostgreSQL-backed training loop for continuous improvement.

## Module Registry

| Module | Directory | Entry Point | Status | AI Context Doc |
|--------|-----------|-------------|--------|---------------|
| **router** | `artemis_final/router/` | `public_api.py::init_router()` | PARTIAL | [router.md](modules/router.md) |
| **load_balancer** | `artemis_final/load_balancer/` | `public_api.py::ArtemisLoadBalancerModule` | PARTIAL | [load_balancer.md](modules/load_balancer.md) |
| **ares** | `artemis_final/ares/` | `evaluation/router_eval_pipeline.py::RouterEvalPipeline` | PLACEHOLDER | [ares.md](modules/ares.md) |
| **inference_engine** | `artemis_final/inference_engine/` | `runners.py::OpenAIStyleRunner` | PLACEHOLDER | [inference_engine.md](modules/inference_engine.md) |
| **router_train** | `artemis_final/router_train/` | `notebooks/02_reward_router_sql_to_training.ipynb` | PLACEHOLDER | [router_train.md](modules/router_train.md) |
| **data_loop** | `artemis_final/data_loop/` | `collector.py::DataCollector` | PLACEHOLDER | [data_loop.md](modules/data_loop.md) |
| **system_api** | `artemis_final/system_api/` | `main.py` (FastAPI app) | COMPLETE | [system_api.md](modules/system_api.md) |
| **common** | `artemis_final/common/` | `config_loader.py::load_global_config()` | COMPLETE | [common.md](modules/common.md) |
| **cascadeflow** | `code_base/cascadeflow/` | `routing/domain.py::DomainCascadeStrategy` | PLACEHOLDER | [cascadeflow.md](modules/cascadeflow.md) |
| **frugal_gpt** | `code_base/frugal_gpt/` | `FrugalGPT/src/service/modelservice.py` | PARTIAL | [frugal_gpt.md](modules/frugal_gpt.md) |
| **lovm** | `code_base/lovm/` | `lovm/lovm/lovm.py::LOVM` | COMPLETE | [lovm.md](modules/lovm.md) |
| **artemis_core/src** | `artemis_core/src/artemis/` | `router/router.py::ArtemisRouter` | COMPLETE | [artemis_core_src.md](modules/artemis_core_src.md) |
| **artemis_core** | `artemis_core/` | `main.py` | PARTIAL | [artemis_core.md](modules/artemis_core.md) |
| **which_vlm/artemis** | `code_base/which_vlm/artemis/` | — (no entry) | PLACEHOLDER | [which_vlm_artemis.md](modules/which_vlm_artemis.md) |
| **which_vlm/ares** | `code_base/which_vlm/ares/` | — (no entry) | PLACEHOLDER | [which_vlm_ares.md](modules/which_vlm_ares.md) |
| **which_vlm/experiments** | `code_base/which_vlm/experiments/` | — (no entry) | COMPLETE | [which_vlm_experiments.md](modules/which_vlm_experiments.md) |
| **helpers** | `code_base/helpers/` | `gpu_metrics.py` | COMPLETE | [helpers.md](modules/helpers.md) |
| **examples** | `examples/` | `examples/README.md` | COMPLETE | [examples.md](modules/examples.md) |
| **aurelio** | `code_base/aurelio/` | Dataset utilities | COMPLETE | [aurelio.md](modules/aurelio.md) |
| **cascade_experiments** | `artemis_final/cascade_experiments/` | Experiment scripts | COMPLETE | [cascade_experiments.md](modules/cascade_experiments.md) |
| **tests** | `artemis_final/tests/` | Test modules | COMPLETE | [tests.md](modules/tests.md) |
| **system_api** | `artemis_final/system_api/` | `main.py` | COMPLETE | [system_api.md](modules/system_api.md) |

Full 42-module listing at [SYSTEM_STATE.md](SYSTEM_STATE.md).

## Data Flow

```
1. Input:  User POSTs to /v1/chat/completions with {messages, router_mode, image?}
              │
2. Router: init_router() → route_request(prompt, mode, metadata)
              │  DistilBERT encode → model/mode embeddings → MLP → reward scores per VLM
              │  Returns: {chosen_model, rewards, mode, inference_ms}
              ▼
3. Load Balancer: schedule(router_output, context)
              │  SLA check + queue capacity check
              │  Override to next-best if preferred model overloaded
              │  Returns: {chosen_model, is_overloaded, est_latency_ms, est_cost_usd}
              ▼
4. Inference Engine: WhichVLMClient.run_image(prompt, image, chosen_model)
              │  OpenAI-compatible POST to VLM backend
              │  Returns: {text, usage, latency_ms, cost}
              ▼
5. Output: Response returned to user; decision logged
              │
6. Async — Evaluation: RouterEvalPipeline runs Scorer + VLMJudge + Glider
              │  Writes results to PostgreSQL
              ▼
7. Async — Retraining: Periodic retrain from accumulated PostgreSQL data
              │  New checkpoint → hot-swap into router
```

## Cross-Module Data Contracts

| Contract | Defined In | Used By | Key Fields |
|---|---|---|---|
| `RouterOutput` | `load_balancer/core/types.py` | LB, System API | sample_id, task_type, router_probs, preferred_model, max_prob |
| `SchedulingDecision` | `load_balancer/core/types.py` | System API, IE | chosen_model, is_overloaded, est_latency_ms, est_cost_usd, queue_delay_ms, sla_violated |
| `Sample` | `router/core/schemas.py` | Router, ARES, DB | sample_id, source, text, image, label, metadata |
| `RouterDecision` | `router/core/schemas.py` | Router, LB | chosen_model, probs, raw_logits, model_order, inference_ms |
| `GlobalConfig` | `common/config_loader.py` | All modules | db.url, router.checkpoint_path, load_balancer config, IE models |
| `TrafficSimulationResult` | `load_balancer/public_api.py` | System API | avg_latency_ms, p95_latency_ms, sla_violation_rate, avg_cost_usd |
| `BudgetExhaustedError` | `load_balancer/core/types.py` | System API | Raised when no model meets constraints → HTTP 503 |

## Critical Warnings for Agents

> **Do not build on these without fixing first:**

| File | Line | Issue | Workaround |
|---|---|---|---|
| `artemis_final/router/core/traffic_simulator.py` | 142 | `raise NotImplementedError` in `run()` | Use `load_balancer::simulate_traffic()` instead |
| `artemis_final/inference_engine/runners.py` | — | `run_batch` and key methods return `False` | Client is not functional |
| `code_base/frugal_gpt/.../modelservice.py` | 59, 117, 122 | `raise NotImplementedError` | Module not runnable |
| `artemis_final/data_loop/retrainer.py` | — | `retrain()` body is empty | Manual retraining via notebooks only |
| `code_base/cascadeflow/.../strategies/domain.py` | — | DomainCascadeStrategy not implemented | Falls back to QualityCascadeStrategy |
| `artemis_final/ares/public_api.py` | 79, 107 | `return None` (silent failure) | Errors pass silently |
| `code_base/which_vlm/artemis` | 16, 27, 35 | All methods return `False` | Use `artemis_final/router/` instead |
| `code_base/which_vlm/ares` | 87, 94, 95 | Multiple `return None` | Use `artemis_final/ares/` instead |

> **Use notebooks for training.** `router_train/service.py` has placeholder returns. Use `notebooks/router_train/02_reward_router_sql_to_training.ipynb` directly.

> **144 placeholder returns** across the codebase. Most concentrated in cascadeflow (55), ares (38), which_vlm/ares (11), and which_vlm/artemis (8). See [SYSTEM_STATE.md](SYSTEM_STATE.md) for the full breakdown.

## Update Protocol

When source code changes, run:

```bash
python scripts/check-doc-drift.py --since HEAD
```

This reads `docs/meta/SCAN_MANIFEST.json` and outputs an ordered list of documentation files that need updating. Follow every item before marking a task complete.

If `SCAN_MANIFEST.json` is out of date (new files added or modified), re-run the discovery scan and update it first:

```bash
# After any significant code changes
python scripts/check-doc-drift.py --since HEAD --rescan
```

See `docs/meta/SCAN_LOG.md` for the full scan summary including notable findings and module status breakdown.
