# Codebase Map: frugal_gpt
>
> Directory: code_base/frugal_gpt/FrugalGPT/
> Entry point: FrugalGPT/src/service/modelservice.py::make_model()
> Status: PARTIAL (module not runnable — NotImplementedError in key methods)

## Responsibility

Multi-provider LLM service with model-chaining strategy for cost minimization. Provides API access to 8 LLM backends via a unified interface, and a strategy optimizer that finds minimum-cost model chains meeting quality thresholds.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `src/service/modelservice.py` | Core | Abstract `ModelService`; concrete providers (OpenAI, AI21, Cohere, Anthropic, Gemini, Together AI); `make_model()` factory |
| `src/service/utils.py` | Utility | `compute_cost()` from token counts |
| `src/orchestration/llmchain.py` | Orchestration | Strategy pattern: `Strategy` base, `SingleAPI` (stub), `LLMChain` (complete) |
| `src/FrugalGPT/optimizer.py` | Core | Data construction + optimization for LLM chain thresholds |
| `src/FrugalGPT/scoring.py` | Core | LLM scoring utilities |
| `src/FrugalGPT/frugalgpt.py` | Runner | Main FrugalLLM class |
| `src/FrugalGPT/llmcache.py` | Core | LLM response caching |
| `src/FrugalGPT/llmvanilla.py` | Runner | Single-model baseline |
| `src/FrugalGPT/llmcascade.py` | Runner | Cascade strategy |
| `src/FrugalGPT/utils.py` | Utility | Shared utilities |
| `src/FrugalGPT/dataloader.py` | Utility | Evaluation data loading |
| `src/FrugalGPT/evaluate.py` | Runner | Evaluation harness |

## Change Guide

- **To add a new API provider**: subclass `APIModelProvider`, implement `_request_format()`, `_response_format()`, `_get_io_tokens()`, `_get_endpoint()`, `_api_call()`; register in `_PROVIDER_MAP`
- **To change the optimization objective**: edit `src/FrugalGPT/optimizer.py::optimize()` — controls threshold and model chain selection
- **To add a new LLM strategy**: subclass `Strategy` in `llmchain.py`; implement `train()` and `predict()`

## Call Chain

```
make_model(provider, model)
  → _PROVIDER_MAP[provider](model)
    → APIModelProvider.__init__()
      → sets _API_KEY from env, _ENDPOINT from env/class var

APIModelProvider.getcompletion(context, genparams)
  → _request_format(context, genparams)
  → _api_call(endpoint, data, api_key)
  → _response_format(response)
  → _get_cost(context, result)
  → returns {completion, raw, cost, latency}

LLMChain.train(responses, labels, scores)
  → _find_param() for each permutation
    → optimizer.construct_data()
    → optimizer.optimize()
  → pick best result
  → savestrategy()

LLMChain.predict(responses, scores)
  → compute_distance_batch() for each chain prefix
  → predict_one() → walk through models, stop at first below threshold
  → returns {answer, cost, query_apis}
```

## Dependencies

Internal: `FrugalGPT.optimizer`, `service.utils`
External: `requests`, `openai`, `anthropic`, `cohere`, `ai21`, `google.generativeai`, `transformers.GPT2Tokenizer`
