# Module: ares
>
> Status: PLACEHOLDER
> Directory: artemis_final/ares/
> Entry point: public_api.py
> Last updated: 2026-06-20

## Purpose

Evaluates VLM responses for quality using ground-truth scoring, VLM Judge (Molmo), and Glider. Writes results to PostgreSQL for downstream router retraining. Also provides data loading and caching utilities.

## Entry Point

`public_api.py` — module facade with init and evaluation helpers. Primary orchestration is `RouterEvalPipeline` in `evaluation/router_eval_pipeline.py`.

## Public API

| Function | File | Purpose |
|---|---|---|
| `RouterEvalPipeline` | `evaluation/router_eval_pipeline.py` | Full eval: scorer + VLM judge + glider, parallel execution, DB writeback |
| `Scorer` | `evaluation/evaluation.py` | Ground-truth-based accuracy/F1 scoring |
| `GliderEvaluator` | `evaluation/evaluation.py` | Text-only fast evaluator; may load heavy models |
| `VLMJudge` | `evaluation/judge_molmo.py` | Listwise ranking with image (Molmo-based) |
| `estimate_confidence` | `evaluation/confidence.py` | Confidence score estimation |
| `insert_evaluations` | `db/operations.py` | Write evaluation results to PostgreSQL |
| `load_per_task_model_stats` | `db/operations.py` | Load historical stats from DB |

## Internal Structure

| File | Layer | Responsibility |
|---|---|---|
| `public_api.py` | Runner | Module facade; init; some stubs return None |
| `evaluation/router_eval_pipeline.py` | Orchestration | Full eval pipeline orchestration with ThreadPoolExecutor |
| `evaluation/evaluation.py` | Core | Scorer (ground truth), GliderEvaluator (text-only), parse_glider_output |
| `evaluation/judge_molmo.py` | Core | VLMJudge — Molmo listwise ranking with image input |
| `evaluation/confidence.py` | Core | estimate_confidence() — confidence scoring |
| `db/operations.py` | Core | SQLAlchemy models (samples, responses, evaluations, images); CRUD ops |
| `db/migrations/` | Schema | Alembic migrations for schema version management |
| `data/dataset_loader.py` | Core | Load samples from DB into DataFrames |
| `imports/cached_dataset.py` | Utility | Cached dataset for fast reloading |
| `configs/` | Config | DB and model configuration |
| `notebooks/` | Runner | 01_parallel_inference_to_db, 02_eval_scoring, 03_cost_utility, 04_eda_evaluations |

## Dependencies

Internal: `common`, `inference_engine`, `router`
External: `sqlalchemy`, `psycopg2`, `transformers` (Molmo), `tqdm`, `concurrent.futures`

## Known Issues

- `public_api.py` — multiple `return None` placeholder returns for error paths. Errors silently pass.
- `evaluation/evaluation.py:32` — NOTE: GliderEvaluator and VLMJudge may load heavy models at init time.
- Data collection path has many stubs; reliability depends on inference_engine being functional.
- RouterEvalPipeline depends on DB being populated and inference_engine working.
- `imports/cached_dataset.py` — cached dataset loading; verify cache validity on updates.

## What an Agent Must Know Before Editing

- VLMJudge uses Molmo for listwise ranking with images — requires GPU memory for the model.
- GliderEvaluator is text-only and faster, but may also load a heavy model.
- The eval pipeline runs in parallel via ThreadPoolExecutor. Adjust `max_workers` based on VLM backend capacity.
- DB schema uses SQLAlchemy. Migrations are managed via Alembic in `db/migrations/`.
- Evaluation results feed directly into the router training pipeline — schema changes to `vlm_evaluations` need corresponding updates to `router_train/db_utils.py`.
