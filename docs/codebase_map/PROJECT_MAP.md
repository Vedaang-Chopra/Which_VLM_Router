# Project Map
>
> Last updated: 2026-06-20 | Source: docs/meta/SCAN_MANIFEST.json

## Directory Index

| Directory | Type | Responsibility | Entry Point |
|---|---|---|---|
| `artemis_final/router/` | Module | VLM routing: DistilBERT + MLP reward prediction | `router/public_api.py::init_router()` |
| `artemis_final/load_balancer/` | Module | SLA-aware capacity scheduling | `load_balancer/public_api.py::ArtemisLoadBalancerModule` |
| `artemis_final/ares/` | Module | Evaluation: Scorer + VLMJudge + Glider; DB ops | `ares/evaluation/router_eval_pipeline.py` |
| `artemis_final/inference_engine/` | Module | OpenAI-compatible VLM inference client | `inference_engine/runners.py::OpenAIStyleRunner` |
| `artemis_final/router_train/` | Module | Router training from PostgreSQL data | `router_train/notebooks/02_reward_router_sql_to_training.ipynb` |
| `artemis_final/data_loop/` | Module | Online logging and periodic retraining | `data_loop/collector.py` |
| `artemis_final/system_api/` | Module | FastAPI app (OpenAI-compatible /v1/chat/completions) | `system_api/main.py` |
| `artemis_final/common/` | Module | Shared config loading and utilities | `common/config_loader.py::load_global_config()` |
| `artemis_core/src/artemis/` | Module | Minimal reference implementation | `artemis_core/src/artemis/router/router.py` |
| `code_base/cascadeflow/` | Research | Cascade routing strategies | `cascadeflow/routing/domain.py` |
| `code_base/frugal_gpt/` | Research | FrugalGPT model selection | `FrugalGPT/src/service/modelservice.py` |
| `code_base/lovm/` | Research | LOVM orchestration and profiling | `lovm/lovm/lovm.py` |
| `code_base/which_vlm/` | Variant | Alternative ARTEMIS implementations | — (use artemis_final instead) |
| `code_base/aurelio/` | Dataset | Pivot dataset utilities | `code_base/aurelio/` |
| `examples/` | Examples | Load balancer, router, and ops examples | `examples/README.md` |
| `artemis_final/checkpoints/` | Data | Trained router `.pt` files | — |
| `artemis_final/configs/` | Config | `artemis.yaml` master config | — |
| `docs/` | Docs | This documentation system | `docs/ai_context/INDEX.md` |

## Top-Level Entry Points

| Command | File | What It Does |
|---|---|---|
| `python -m uvicorn system_api.main:app` | `system_api/main.py` | Start FastAPI server (production) |
| `python artemis_core/main.py` | `artemis_core/main.py` | Run minimal ARTEMIS pipeline |
| `docker-compose up` | `docker-compose.yml` | Full stack: FastAPI + PostgreSQL |
| `jupyter notebook notebooks/` | notebooks/ | Training and evaluation notebooks |
| `bash artemis_final/scripts/run_demo.sh` | `artemis_final/scripts/run_demo.sh` | Full pipeline demo |

## Cross-Module Data Contracts

| Contract | Defined In | Used By | Key Fields |
|---|---|---|---|
| `RouterOutput` | `load_balancer/core/types.py` | LB, System API | sample_id, task_type, router_probs, preferred_model, max_prob |
| `SchedulingDecision` | `load_balancer/core/types.py` | System API, inference | chosen_model, is_overloaded, est_latency_ms, est_cost_usd, queue_delay_ms |
| `Sample` | `router/core/schemas.py` | Router, ARES, DB | sample_id, source, text, image, label, metadata |
| `RouterDecision` | `router/core/schemas.py` | Router, LB | chosen_model, probs, raw_logits, model_order, inference_ms |
| `GlobalConfig` | `common/config_loader.py` | All modules | db.url, router.checkpoint_path, load_balancer config, inference_engine models |
| `TrafficSimulationResult` | `load_balancer/public_api.py` | System API | avg_latency_ms, p95_latency_ms, sla_violation_rate, avg_cost_usd |

## External Model and Service Dependencies

| Dependency | Version | Used By | Purpose |
|---|---|---|---|
| DistilBERT | `distilbert-base-uncased` | router | Frozen text encoder (66M params) |
| VLM Backends | API | inference_engine | 5 VLMs via OpenAI-compatible API |
| PostgreSQL | 14+ | ares, router_train | Sample/response/evaluation storage |
| Molmo | latest | ares | VLM Judge for listwise ranking |
| Glider | latest | ares | Text-only fast evaluator |
| W&B | — | router | Optional: experiment tracking |

## Config Files

| File | What It Controls | Format |
|---|---|---|
| `artemis_final/configs/artemis.yaml` | Master config: DB, router checkpoint, LB settings | YAML |
| `artemis_final/router/router_config_reward.yaml` | Router MLP dims, model list, embedding sizes | YAML |
| `artemis_final/load_balancer/core/config.py` | Default capacity config | Python |
| `artemis_final/ares/configs/models.yaml` | VLM backend URLs, API keys, model types | YAML |
| `artemis_final/router_train/data/model_index.json` | Model name → index mapping | JSON |
| `artemis_final/router_train/data/mode_index.json` | Mode name → index mapping | JSON |
| `artemis_final/router_train/data/task_index.json` | Task type → index mapping | JSON |
