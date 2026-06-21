# Codebase Map: inference_engine
>
> Directory: artemis_final/inference_engine/
> Entry point: runners.py::OpenAIStyleRunner
> Status: PLACEHOLDER

## Responsibility

Unified OpenAI-compatible client for calling all five VLM backends. Handles image+text and text-only inference, extracts token usage and cost, tracks latency.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `runners.py` | Runner | `OpenAIStyleRunner` — batch inference runner (PARTIAL: methods return False) |
| `inference_service.py` | Orchestration | `InferenceService` — service-level orchestration (PARTIAL) |
| `client.py` | Core | `WhichVLMClient` — LLM + VLM sub-clients via OpenAI-compatible API |
| `messages.py` | Schema | Message formatting; `return False` at lines 70, 73 |
| `models.py` | Schema | Model metadata dataclasses |
| `config.py` | Schema | InferenceConfig loading |
| `suites.py` | Utility | Test suites |
| `readme.md` | Doc | API reference |

## Change Guide

- **To fix client**: complete `runners.py::run_batch()` and `inference_service.py` methods — these are blocking stubs
- **To add a new VLM**: add to `ares/configs/models.yaml` and ensure OpenAI-compatible endpoint
- **To change image handling**: edit `client.py::VLMClient._encode_image()` for PIL or base64

## Dependencies

External: `openai` (OpenAI SDK), `requests`, `PIL`, `base64`, `numpy`
