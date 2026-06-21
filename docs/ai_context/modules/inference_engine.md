# Module: inference_engine
>
> Status: PLACEHOLDER
> Directory: artemis_final/inference_engine/
> Entry point: runners.py::OpenAIStyleRunner
> Last updated: 2026-06-20

## Purpose

Unified OpenAI-compatible client for calling all five VLM backends. Handles image+text (VLM) and text-only (LLM) inference, extracts token usage and cost, and tracks latency.

## Entry Point

`runners.py` — `OpenAIStyleRunner` class. Primary service is `inference_service.py::InferenceService`.

## Public API

| Function | File | Purpose |
|---|---|---|
| `OpenAIStyleRunner` | `runners.py` | Batch and single inference runner (PARTIAL: some methods return False) |
| `InferenceService` | `inference_service.py` | Service-level inference orchestration |

## Internal Structure

| File | Layer | Responsibility |
|---|---|---|
| `runners.py` | Runner | OpenAIStyleRunner; batch run methods |
| `inference_service.py` | Runner | InferenceService; service entry |
| `client.py` | Core | WhichVLMClient — LLM + VLM sub-clients, OpenAI-compatible |
| `messages.py` | Schema | Message formatting for API calls |
| `models.py` | Schema | Model metadata dataclasses |
| `config.py` | Schema | InferenceConfig loading |
| `suites.py` | Core | Test suites for inference validation |
| `readme.md` | Doc | API reference documentation |

## Dependencies

External: `openai`, `requests`, `PIL`, `base64`, `time`

## Known Issues

- `runners.py` — key batch methods return `False` (incomplete).
- `inference_service.py:44,50` — `return None` placeholders.
- `messages.py:70,73` — `return False` placeholders.
- This module is not functional. It cannot run inference until these are completed.

## What an Agent Must Know Before Editing

- The client targets OpenAI-compatible backends. All five VLMs must expose `/v1/chat/completions`.
- Token usage is extracted from the response `usage` field — backends must return this.
- Image handling: `client.py` accepts PIL Image or base64-encoded image data.
- This module is a dependency for ARES (evaluation needs inference) and the System API.
