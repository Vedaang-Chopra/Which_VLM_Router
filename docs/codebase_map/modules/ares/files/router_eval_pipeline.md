# router_eval_pipeline.py
>
> Module: ares
> Layer: Orchestration
> Path: artemis_final/ares/evaluation/router_eval_pipeline.py

## Purpose

Full evaluation pipeline: orchestrates Scorer + VLMJudge (Molmo) + GliderEvaluator across all samples and models in parallel using ThreadPoolExecutor, then writes results to PostgreSQL.

## Key Functions

| Function | Signature | What it does |
|---|---|---|
| run_evaluation | `run_evaluation(samples, models)` | Main pipeline; parallel eval + DB writeback |

## Imports

Internal: `ares.evaluation.evaluation`, `ares.evaluation.judge_molmo`, `ares.evaluation.confidence`, `ares.db.operations`, `inference_engine.runners`
External: `sqlalchemy`, `tqdm`, `concurrent.futures`, `pandas`

## Known Issues

Depends on working `inference_engine`. If inference_engine returns False, evaluation cannot run. Molmo and Glider load heavy models at init.
