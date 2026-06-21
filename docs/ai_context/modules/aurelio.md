# Module: aurelio
>
> Status: COMPLETE
> Directory: code_base/aurelio/
> Last updated: 2026-06-21

## Purpose

Provides pivot dataset utilities (train and test parquet files) for router training and evaluation.

## Internal Structure

| File | What it does |
|---|---|
| `router_pivot_dataset_train.parquet` | Training split: sample_id, prompt_raw, ground_truth, router_task, model scores |
| `router_pivot_dataset_test.parquet` | Test split: same schema as train |
| Dataset loading utilities | Load and split data for training/evaluation |

## Known Issues

None. Both parquet files are clean.

## What an agent must know before editing this module

- These are static data files, not code. No module dependencies.
- The schema includes: `sample_id`, `prompt_raw`, `ground_truth`, `router_task`, and per-model scoring columns.
- Used for offline router evaluation and as a quick-start alternative to the full PostgreSQL pipeline.
