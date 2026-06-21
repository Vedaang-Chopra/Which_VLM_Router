# Module: Inference Engine

## What It Does

Unified OpenAI-compatible client for calling all five VLM backends. Handles image+text requests, token usage extraction, latency tracking, and cost computation.

## Key Files

| File | What It Does |
|---|---|
| `runners.py` | OpenAIStyleRunner skeleton; batch run methods |
| `inference_service.py` | Service-level inference entry |
| `client.py` | WhichVLMClient — unified LLM and VLM interface |

## Status

**PLACEHOLDER.** The client structure exists (WhichVLMClient with LLM/VLM sub-clients) but `run_batch` and key methods return `False`, indicating incomplete implementation. Depends on `inference_engine/` being functional for the full pipeline to work.
