# Module Map: ares
>
> Directory: artemis_final/ares/
> Entry point: evaluation/router_eval_pipeline.py::RouterEvalPipeline
> Status: PLACEHOLDER

## Responsibility

Evaluates VLM response quality using ground truth scoring, VLM Judge (Molmo), and Glider. Writes all results to PostgreSQL for downstream router retraining.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `public_api.py` | Runner | Module facade (partial stubs) |
| `evaluation/router_eval_pipeline.py` | Orchestration | Full eval orchestration with ThreadPoolExecutor |
| `evaluation/evaluation.py` | Core | Scorer (ground truth), GliderEvaluator, parse_glider_output |
| `evaluation/judge_molmo.py` | Core | VLMJudge — Molmo listwise ranking with image |
| `evaluation/confidence.py` | Core | estimate_confidence() confidence scoring |
| `db/operations.py` | Core | SQLAlchemy models + CRUD operations |
| `db/migrations/` | Schema | Alembic migration scripts |
| `data/dataset_loader.py` | Core | Load samples from DB |
| `notebooks/` | Runner | Evaluation workflow notebooks |

## Public API

| Function | File | Signature | Purpose |
|---|---|---|---|
| `RouterEvalPipeline` | evaluation/router_eval_pipeline.py | `__init__(config, db_session)` | Init eval pipeline |
| `run_evaluation` | evaluation/router_eval_pipeline.py | `run_evaluation(samples, models)` | Run full eval |
| `Scorer` | evaluation/evaluation.py | `score(response, ground_truth)` | Ground truth accuracy/F1 |
| `VLMJudge` | evaluation/judge_molmo.py | `judge(samples, responses)` | Molmo listwise ranking |
| `estimate_confidence` | evaluation/confidence.py | `estimate_confidence(response)` | Confidence score |
| `insert_evaluations` | db/operations.py | `insert_evaluations(session, evals)` | Write to DB |

## Change Guide

- **To add a new evaluator**: implement in `evaluation/` and add to `RouterEvalPipeline`
- **To change DB schema**: add Alembic migration in `db/migrations/`
- **To scale evaluation**: adjust `max_workers` in ThreadPoolExecutor
