# operations.py
>
> Module: ares
> Layer: Core
> Path: artemis_final/ares/db/operations.py

## Purpose

SQLAlchemy models and CRUD operations for all PostgreSQL tables: vlm_samples, vlm_images, vlm_responses, vlm_evaluations. Reads and writes the ARES database.

## Key Functions

| Function | Signature | What it does |
|---|---|---|
| insert_evaluations | `insert_evaluations(session, evals)` | Bulk insert evaluation results |
| load_per_task_model_stats | `load_per_task_model_stats(path)` | Load historical stats from JSON |

## Imports

Internal: None
External: `sqlalchemy`, `psycopg2`

## Known Issues

Schema migrations are managed via Alembic in `db/migrations/`. Changing table structures requires a migration.
