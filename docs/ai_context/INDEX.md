# AI Context Index — ARTEMIS
>
> Last updated: 2026-06-21
> Full scan: docs/meta/SCAN_MANIFEST.json (521 files scanned)
> System state: docs/ai_context/SYSTEM_STATE.md
> 42 manifest entries | 28 COMPLETE, 4 PARTIAL, 10 PLACEHOLDER
## What This System Does

ARTEMIS is a cost-aware Vision-Language Model (VLM) router that uses frozen CLIP + DistilBERT encoders and an MLP classifier to dispatch multimodal queries to the cheapest model meeting accuracy constraints, routing across five VLMs: deepseek_ocr, qwen2_5_vl_3b, qwen2_5_vl_7b, qwen3_vl_8b_thinking, and gemma_3_27b. It supports four routing modes (accuracy, cheap, fast, balanced) and uses a PostgreSQL-backed training loop for continuous improvement.

## Module Registry

Manifest names are preserved below. Nested names use underscores only in documentation filenames.

| Module | Files | Entry Point | Status | AI Context Doc |
|---|---:|---|---|---|
| **cascadeflow** | 59 | `code_base/cascadeflow/cascadeflow/cascadeflow/routing/domain.py` | PLACEHOLDER | [cascadeflow.md](modules/cascadeflow.md) |
| **ares** | 52 | `artemis_final/ares/public_api.py` | PLACEHOLDER | [ares.md](modules/ares.md) |
| **which_vlm/artemis** | 31 | Not identified by the manifest | PLACEHOLDER | [which_vlm_artemis.md](modules/which_vlm_artemis.md) |
| **lovm** | 27 | Not identified by the manifest | COMPLETE | [lovm.md](modules/lovm.md) |
| **router** | 20 | `artemis_final/router/public_api.py` | PLACEHOLDER | [router.md](modules/router.md) |
| **frugal_gpt** | 19 | Not identified by the manifest | PARTIAL | [frugal_gpt.md](modules/frugal_gpt.md) |
| **router_train** | 19 | Not identified by the manifest | PLACEHOLDER | [router_train.md](modules/router_train.md) |
| **root** | 17 | Not identified by the manifest | PLACEHOLDER | [root.md](modules/root.md) |
| **which_vlm/experiments** | 16 | Not identified by the manifest | COMPLETE | [which_vlm_experiments.md](modules/which_vlm_experiments.md) |
| **artemis_core/src** | 14 | Not identified by the manifest | COMPLETE | [artemis_core_src.md](modules/artemis_core_src.md) |
| **load_balancer** | 14 | `artemis_final/load_balancer/public_api.py` | PARTIAL | [load_balancer.md](modules/load_balancer.md) |
| **which_vlm/ares** | 12 | Not identified by the manifest | PLACEHOLDER | [which_vlm_ares.md](modules/which_vlm_ares.md) |
| **inference_engine** | 9 | `artemis_final/inference_engine/runners.py` | PLACEHOLDER | [inference_engine.md](modules/inference_engine.md) |
| **common** | 7 | Not identified by the manifest | COMPLETE | [common.md](modules/common.md) |
| **examples/load_balancer** | 7 | Not identified by the manifest | COMPLETE | [examples_load_balancer.md](modules/examples_load_balancer.md) |
| **which_vlm/inference_api_call** | 7 | `code_base/which_vlm/inference_api_call/runners.py` | PLACEHOLDER | [which_vlm_inference_api_call.md](modules/which_vlm_inference_api_call.md) |
| **data_loop** | 5 | Not identified by the manifest | PLACEHOLDER | [data_loop.md](modules/data_loop.md) |
| **which_vlm/configs** | 5 | Not identified by the manifest | COMPLETE | [which_vlm_configs.md](modules/which_vlm_configs.md) |
| **system_api** | 4 | `artemis_final/system_api/main.py` | COMPLETE | [system_api.md](modules/system_api.md) |
| **examples/router** | 3 | Not identified by the manifest | COMPLETE | [examples_router.md](modules/examples_router.md) |
| **README.md** | 2 | Not identified by the manifest | COMPLETE | [README.md.md](modules/README.md.md) |
| **aurelio** | 2 | Not identified by the manifest | COMPLETE | [aurelio.md](modules/aurelio.md) |
| **examples/ops** | 2 | Not identified by the manifest | COMPLETE | [examples_ops.md](modules/examples_ops.md) |
| **helpers** | 2 | Not identified by the manifest | COMPLETE | [helpers.md](modules/helpers.md) |
| **main.py** | 2 | `artemis_final/main.py` | COMPLETE | [main.py.md](modules/main.py.md) |
| **requirements.txt** | 2 | Not identified by the manifest | COMPLETE | [requirements.txt.md](modules/requirements.txt.md) |
| **tests** | 2 | Not identified by the manifest | COMPLETE | [tests.md](modules/tests.md) |
| **01_router_single_and_batch_modes.ipynb** | 1 | Not identified by the manifest | COMPLETE | [01_router_single_and_batch_modes.ipynb.md](modules/01_router_single_and_batch_modes.ipynb.md) |
| **02_router_experiments_and_modes.ipynb** | 1 | Not identified by the manifest | COMPLETE | [02_router_experiments_and_modes.ipynb.md](modules/02_router_experiments_and_modes.ipynb.md) |
| **COMPLETE_SYSTEM_OVERVIEW.md** | 1 | Not identified by the manifest | PARTIAL | [COMPLETE_SYSTEM_OVERVIEW.md.md](modules/COMPLETE_SYSTEM_OVERVIEW.md.md) |
| **IMPLEMENTATION_WALKTHROUGH.md** | 1 | Not identified by the manifest | COMPLETE | [IMPLEMENTATION_WALKTHROUGH.md.md](modules/IMPLEMENTATION_WALKTHROUGH.md.md) |
| **REFACTOR_NOTES.md** | 1 | Not identified by the manifest | COMPLETE | [REFACTOR_NOTES.md.md](modules/REFACTOR_NOTES.md.md) |
| **artemis_core** | 1 | Not identified by the manifest | PARTIAL | [artemis_core.md](modules/artemis_core.md) |
| **artemis_final** | 1 | Not identified by the manifest | COMPLETE | [artemis_final.md](modules/artemis_final.md) |
| **cascade_experiments** | 1 | Not identified by the manifest | COMPLETE | [cascade_experiments.md](modules/cascade_experiments.md) |
| **config** | 1 | Not identified by the manifest | COMPLETE | [config.md](modules/config.md) |
| **docker-compose.yml** | 1 | Not identified by the manifest | COMPLETE | [docker-compose.yml.md](modules/docker-compose.yml.md) |
| **examples** | 1 | Not identified by the manifest | COMPLETE | [examples.md](modules/examples.md) |
| **examples/README.md** | 1 | Not identified by the manifest | COMPLETE | [examples_README.md.md](modules/examples_README.md.md) |
| **test_vllm_model.ipynb** | 1 | Not identified by the manifest | COMPLETE | [test_vllm_model.ipynb.md](modules/test_vllm_model.ipynb.md) |
| **utils** | 1 | Not identified by the manifest | COMPLETE | [utils.md](modules/utils.md) |
| **which_vlm/__init__.py** | 1 | Not identified by the manifest | COMPLETE | [which_vlm___init__.py.md](modules/which_vlm___init__.py.md) |

The manifest is the coverage source of truth. Statuses follow its scan except where [SYSTEM_STATE.md](SYSTEM_STATE.md) has a more conservative component-level result; this applies to the PARTIAL `artemis_core` wrapper.
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
