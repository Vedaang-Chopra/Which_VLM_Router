# Module: frugal_gpt
>
> Status: PARTIAL
> Entry point: FrugalGPT/src/service/modelservice.py::ModelService
> Last updated: 2026-06-21

## Purpose

Decides whether to use a local model, a single API model, or a chain of LLMs to minimize cost while meeting a quality threshold. Contains both the FrugalLLM orchestration logic and the multi-provider API service layer.

## Public API

| Class | File | Signature | What it does |
|---|---|---|---|
| `ModelService` | `modelservice.py` | `__init__(model)` | Abstract base for all API providers |
| `APIModelProvider` | `modelservice.py` | `getcompletion(context, ...)` | REST API provider with retry logic |
| `OpenAIModelProvider` | `modelservice.py` | subclass | OpenAI completion API |
| `OpenAIChatModelProvider` | `modelservice.py` | subclass | OpenAI chat API |
| `AI21ModelProvider` | `modelservice.py` | subclass | AI21 Studio API |
| `CohereAIModelProvider` | `modelservice.py` | subclass | Cohere API |
| `AnthropicModelProvider` | `modelservice.py` | subclass | Anthropic API |
| `GoogleModelProvider` | `modelservice.py` | subclass | Google Gemini API |
| `TogetherAIModelProvider` | `modelservice.py` | subclass | Together AI API |
| `make_model` | `modelservice.py` | `make_model(provider, model)` | Factory function |
| `Strategy` | `llmchain.py` | base class | Abstract training + prediction strategy |
| `LLMChain` | `llmchain.py` | `train()`, `predict()` | Chain-of-thought LLM strategy |

## Internal Structure

| File | Layer | What it does |
|---|---|---|
| `FrugalGPT/src/service/modelservice.py` | Core | All ModelProvider classes (OpenAI, AI21, Cohere, Anthropic, Gemini, Together AI) |
| `FrugalGPT/src/service/utils.py` | Utility | `compute_cost()` for token-based billing |
| `FrugalGPT/src/orchestration/llmchain.py` | Orchestration | Strategy pattern: Strategy, SingleAPI, LLMChain |
| `FrugalGPT/src/FrugalGPT/optimizer.py` | Core | `construct_data()`, `optimize()`, `compute_distance_batch()` |
| `FrugalGPT/src/FrugalGPT/scoring.py` | Core | LLM scoring utilities |
| `FrugalGPT/src/FrugalGPT/frugalgpt.py` | Runner | Main FrugalLLM orchestration |
| `FrugalGPT/src/FrugalGPT/llmcache.py` | Core | LLM response caching |
| `FrugalGPT/src/FrugalGPT/llmvanilla.py` | Runner | Baseline single-LLM approach |
| `FrugalGPT/src/FrugalGPT/utils.py` | Utility | Shared utilities |
| `FrugalGPT/src/FrugalGPT/dataloader.py` | Utility | Data loading for evaluation |
| `FrugalGPT/src/FrugalGPT/evaluate.py` | Runner | Evaluation harness |

## External Dependencies

- `requests` — REST API calls
- `openai` — OpenAI API
- `anthropic` — Anthropic API
- `cohere` — Cohere API
- `ai21` — AI21 Studio API
- `google.generativeai` — Google Gemini
- `transformers.GPT2Tokenizer` — token counting
- `scipy` — optimization (optimizer.py)

## Known Issues

- `FrugalGPT/src/orchestration/llmchain.py:19` — `Strategy.train()` raises `NotImplementedError`. The base Strategy class is abstract.
- `FrugalGPT/src/orchestration/llmchain.py:58` — `SingleAPI.train()` returns `None` (empty body, stub).
- `FrugalGPT/src/service/modelservice.py:59` — `ModelService.getcompletion()` raises `NotImplementedError`.
- `FrugalGPT/src/service/modelservice.py:117` — `APIModelProvider._request_format()` raises `NotImplementedError` (abstract base method).
- `FrugalGPT/src/service/modelservice.py:122` — `APIModelProvider._response_format()` raises `NotImplementedError` (abstract base method).
- **Module is NOT runnable** — all API providers inherit from `ModelService` and must override `getcompletion()`, `_request_format()`, and `_response_format()` before use.

## What an Agent Must Know Before Editing

- `APIModelProvider` is the abstract base class. All concrete providers (OpenAI, AI21, etc.) must implement `_request_format()`, `_response_format()`, and `_get_io_tokens()`.
- The module requires API keys as environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AI21_STUDIO_API_KEY`, `COHERE_STUDIO_API_KEY`, `TOGETHER_API_KEY`, `FOREFRONT_API_KEY`.
- The LLMChain strategy enumerates all permutations of API chains up to `L_max` and optimizes thresholds — expensive for large `L_max`.
- Cost is computed via `service/utils.py::compute_cost()` based on input/output token counts from `_get_io_tokens()`.
- `_PROVIDER_MAP` at the bottom of `modelservice.py` registers all providers. Add new ones here.
- All API calls include retry logic: 10 attempts with 10s grace time before `TimeoutError`.
- The old OpenAI completion endpoint is used (`/v1/engines/{engine}/completions`), not the modern `/v1/chat/completions` — `OpenAIChatModelProvider` uses the chat endpoint instead.
