# Module: system_api
>
> Status: PARTIAL
> Directory: artemis_final/system_api/
> Entry point: main.py (FastAPI application)
> Last updated: 2026-06-20

## Purpose

FastAPI application exposing an OpenAI-compatible `/v1/chat/completions` endpoint. Orchestrates: Router → Load Balancer → Inference Engine and logs all decisions.

## Public API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | Main inference endpoint; runs router → LB → IE |
| `/feedback` | POST | Submit user feedback for retraining |
| `/admin/retrain` | POST | Trigger retraining (stub) |

## Internal Structure

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app with lifespan, routes, error handling |
| `pipeline.py` | Orchestrates Router → LB → IE for each request |
| `schemas.py` | Pydantic request/response models |

## Dependencies

Internal: `router`, `load_balancer`, `inference_engine`
External: `fastapi`, `uvicorn`, `pydantic`

## Known Issues

- Partial implementation; depends on router, load_balancer, and inference_engine all being functional.
- Admin endpoints are stubs.

## What an Agent Must Know

- This is the entry point for production deployments. See `docker-compose.yml` for full stack.
- Requests go through router first, then load balancer, then inference engine. Each step can raise an exception that this module must handle.
- `BudgetExhaustedError` from load_balancer should return HTTP 503 to the client.
