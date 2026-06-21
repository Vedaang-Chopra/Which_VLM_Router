# Codebase Map: system_api
>
> Directory: artemis_final/system_api/
> Entry point: main.py::app (FastAPI)
> Status: COMPLETE

## Responsibility

FastAPI application exposing an OpenAI-compatible `/v1/chat/completions` endpoint. Wires together Router, Load Balancer, Inference Engine, and DataCollector via `pipeline.py`.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `main.py` | Runner | FastAPI app with lifespan; defines `/health`, `/v1/chat/completions`, `/feedback`, `/admin/retrain` |
| `pipeline.py` | Orchestration | `init_system()` wires all services; `handle_chat_completion()` orchestrates Router→LB→IE; `handle_feedback()` stores feedback; `trigger_retrain()` stubs retraining |
| `schemas.py` | Schema | Pydantic models: ChatCompletionRequest/Response, FeedbackRequest/Response, RetrainResponse |

## Change Guide

- **To add a new endpoint**: add to `main.py::app` using `@app.post()` or `@app.get()`
- **To change pipeline orchestration**: edit `pipeline.py::handle_chat_completion()`
- **To handle a new exception type**: add to `main.py::chat_completions()` try/except block

## Call Chain

```
POST /v1/chat/completions
  → main.py::chat_completions()
    → pipeline.py::handle_chat_completion(req, SERVICES)
      → RouterService.route_request(prompt, mode)
      → LoadBalancerService.schedule(router_output)
      → InferenceService.run_image(prompt, image, model)
      → DataCollector.log_sample + log_response
```

## Dependencies

Internal: `router`, `load_balancer`, `inference_engine`, `data_loop`, `common`
External: `fastapi`, `uvicorn`, `pydantic`
