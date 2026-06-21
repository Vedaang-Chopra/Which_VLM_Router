# modelservice.py
>
> Module: frugal_gpt
> Layer: Core
> Path: code_base/frugal_gpt/FrugalGPT/src/service/modelservice.py

## Purpose

Provides API access to 8 LLM backends via a unified `getcompletion()` interface. Each provider handles request formatting, API calls with retry logic, response parsing, and cost computation.

## Classes

### GenerationParameter

Container for generation hyperparameters.

| Field | Default |
|---|---|
| `max_tokens` | 100 |
| `temperature` | 0.1 |
| `stop` | `["\n"]` |
| `date` | "20230301" |

Methods: `get_dict()` — returns readable dict.

### ModelService

Abstract base class for all model providers.

Methods:

- `getcompletion(context, use_save?, savepath?, genparams?) -> dict` — raises `NotImplementedError` at line 59
- `read_response(path)` — read cached pickle response
- `write_response(path)` — write response to pickle

### APIModelProvider(ModelService)

REST API provider with retry logic.

Methods:

- `_request_format(context, genparams)` — **raises NotImplementedError at line 117** (abstract)
- `_response_format(response)` — **raises NotImplementedError at line 122** (abstract)
- `_api_call(endpoint, data, api_key, retries?, retry_grace_time?)` — HTTP POST with 10 retries, 10s grace
- `_get_cost(context, completion)` — calls `_get_io_tokens()` then `compute_cost()`
- `_get_endpoint()` — formats `_ENDPOINT` class var with engine

### Concrete Providers

| Class | Endpoint | Notes |
|---|---|---|
| `OpenAIModelProvider` | `https://api.openai.com/v1/engines/{engine}/completions` | Legacy completion; requires `OPENAI_API_KEY` |
| `OpenAIChatModelProvider` | `https://api.openai.com/v1/chat/completions` | Chat completions; requires `OPENAI_API_KEY` |
| `AI21ModelProvider` | `https://api.ai21.com/studio/v1/{engine}/complete` | Requires `AI21_STUDIO_API_KEY` |
| `CohereAIModelProvider` | Cohere generate API | Requires `COHERE_STUDIO_API_KEY` |
| `ForeFrontAIModelProvider` | Custom fore-front endpoints | Requires `FOREFRONT_API_KEY` |
| `TextSynthModelProvider` | `https://api.textsynth.com/v1/engines/{engine}/completions` | Requires `TEXTSYNTH_API_SECRET_KEY` |
| `AnthropicModelProvider` | `https://api.anthropic.com/v1/complete` | Requires `ANTHROPIC_API_KEY` |
| `GoogleModelProvider` | Google Generative AI | Requires `GEMINI_API_KEY` |
| `TogetherAIModelProvider` | `https://api.together.ai/v1/chat/completions` | Requires `TOGETHER_API_KEY` |

## Key Functions

| Function | Signature | What it does |
|---|---|---|
| `make_model` | `(provider, model)` | Factory: returns provider instance from `_PROVIDER_MAP` |

## Imports

Internal: `service.utils` (compute_cost)
External: `requests`, `openai`, `anthropic`, `cohere`, `ai21`, `google.generativeai`, `transformers.GPT2Tokenizer`

## Known Issues

- Line 59: `ModelService.getcompletion()` raises `NotImplementedError` — all concrete classes must override this
- Line 117: `APIModelProvider._request_format()` raises `NotImplementedError` — all subclasses must implement
- Line 122: `APIModelProvider._response_format()` raises `NotImplementedError` — all subclasses must implement
- API keys read from environment variables at import time — `AssertionError` if not set
- The deprecated `/v1/engines/{engine}/completions` endpoint is used for OpenAI (should use `/v1/chat/completions`)
- `CohereAIModelProvider` has a fallback `except` block that calls with `"test"` prompt — will return wrong results on API failure
