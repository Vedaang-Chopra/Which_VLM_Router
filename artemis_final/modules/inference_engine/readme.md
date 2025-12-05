

# Which_VLM Router – Inference API Call Layer

This package (`which_vlm.inference_api_call`) is a small, focused framework that lets you:

- Configure many **OpenAI-compatible models** (LLMs and VLMs) from a single YAML/JSON file.
- Talk to all of them through a **unified Python interface**.
- Run:
  - Single text prompts across one or many LLMs.
  - Single image(+text) prompts across one or many VLMs.
  - Batch experiments for both LLMs and VLMs.
- Track **latency, usage, and rough cost estimates** for each call.

It is designed to be:

- **Notebook-friendly** – easy to use from Jupyter for quick experiments.
- **Backend-ready** – clean, modular structure suitable for integration into a larger router or evaluation pipeline.
- **Model-agnostic** – as long as your server speaks the OpenAI Chat Completions API (vLLM, LM Studio, etc.), it should work.

---

## High-Level Architecture

At a high level, the package is structured as:

```text
which_vlm/inference_api_call/
  ├── config.py     # Load and define model configurations (from YAML/JSON)
  ├── messages.py   # Build OpenAI-style multimodal messages (text + images)
  ├── models.py     # Small helpers for working with collections of models
  ├── runners.py    # Generic OpenAI-style runner (single + fan-out calls)
  ├── suites.py     # LLMTestSuite & VLMTestSuite (high-level testing helpers)
  └── client.py     # WhichVLMClient – main entrypoint for notebooks and scripts
```

You **almost always** interact with the code through `WhichVLMClient` (`client.py`), which exposes two main handles:

- `client.llm` → **LLMTestSuite** (text-only tests)
- `client.vlm` → **VLMTestSuite** (image+text tests)

Everything else is the plumbing behind those.

---

## Installation & Environment

Make sure you have these dependencies installed in your environment:

```bash
pip install openai pyyaml pillow numpy
```

Also ensure:

- `which_vlm/` and `which_vlm/inference_api_call/` both contain an `__init__.py` so Python treats them as packages.
- Your notebook / script can import the package, e.g.:

```python
import sys, os
sys.path.append(os.path.abspath("."))  # adjust to your repo root
```

If you package this project properly, you can also `pip install -e .` and import directly.

---

## 1. Configuration Layer (`config.py`)

### Purpose

This module defines the **configuration data model** for each model endpoint and provides helpers to load configurations from **YAML** or **JSON** files. It is the bridge between static config files and Python objects.

### Key Dataclass: `ModelEndpoint`

```python
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class ModelEndpoint:
    name: str
    base_url: str
    api_key: str
    model_id: str
    pricing: Dict[str, float]
    extra_params: Dict[str, Any]
    model_type: str = "llm"  # "llm", "vlm", or "both"
```

**Fields:**

- `name`  
  Short logical name (e.g. `"llama-8b"`). This is what you use when calling the suite or runner.

- `base_url`  
  Base URL of the OpenAI-compatible server, e.g.:
  - `http://localhost:8000/v1`
  - `http://10.0.0.5:9000/v1`

- `api_key`  
  API key or token to use for that endpoint. For many local deployments, this can be `"EMPTY"` or any non-empty string.

- `model_id`  
  The exact `model` value expected by the server, e.g.:
  - `"meta-llama-3-8b-instruct"`
  - `"Qwen/Qwen2.5-VL-7B-Instruct"`

- `pricing`  
  Free-form dict, typically:

  ```python
  {
      "prompt_per_1k": 0.0,
      "completion_per_1k": 0.0,
  }
  ```

  Used by the runner for rough cost estimation.

- `extra_params`  
  Extra parameters that should always be sent for this model (merged into the chat payload).  
  Examples:
  - `{"max_tokens": 1024}`
  - `{"top_p": 0.9}`

  ⚠ **Important:** do **not** put `temperature` here; the runner already sets `temperature` explicitly and a duplicate key will cause a Python error.

- `model_type`  
  Used to decide whether a model is treated as an LLM or VLM by higher-level suites:
  - `"llm"` → only in LLM suite
  - `"vlm"` → only in VLM suite
  - `"both"` → appears in both suites

### Loading Models

#### `load_models_from_yaml(path)`

```python
from which_vlm.inference_api_call.config import load_models_from_yaml

models = load_models_from_yaml("models.yaml")
```

- Expects a YAML with a top-level `models` list.

#### `load_models_from_json(path)`

```python
from which_vlm.inference_api_call.config import load_models_from_json

models = load_models_from_json("models.json")
```

Same structure as YAML but in JSON.

### Splitting by Type: `split_models_by_type(models)`

```python
from which_vlm.inference_api_call.config import split_models_by_type

llm_names, vlm_names = split_models_by_type(models)
```

- Reads each model’s `model_type` and returns two lists:
  - `llm_names` (all `"llm"` + `"both"`)
  - `vlm_names` (all `"vlm"` + `"both"`)

---

## 2. Message & Image Utilities (`messages.py`)

### Purpose

Central place for building **OpenAI-style `messages` payloads**, including rich **multimodal** content (text + images).

### Core Types

```python
from typing import Union

ImageLike = Union[str, bytes, "Image.Image", "np.ndarray"]
```

Supported `ImageLike` values:

- `str`:
  - File path (e.g. `"cat.png"`)
  - HTTP/HTTPS URL
  - `data:image/...;base64,...`
  - Bare base64 string
- `bytes` / `bytearray`
- `PIL.Image.Image`
- `numpy.ndarray`

### Helper: `_image_to_part(img)`

Converts an `ImageLike` into an OpenAI image content part:

```python
{
    "type": "image_url",
    "image_url": {"url": "..."}
}
```

Depending on the input, the URL might be:

- The original HTTP/HTTPS URL.
- The original data URL.
- A generated data URL from base64 PNG bytes.

### Main API: `build_messages(...)`

```python
from which_vlm.inference_api_call.messages import build_messages

messages = build_messages(
    prompt: Optional[str] = None,
    images: Optional[Union[Dict[str, Any], List[ImageLike], ImageLike]] = None,
    content_parts: Optional[List[Dict[str, Any]]] = None,
    system: Optional[str] = None,
)
```

#### Behaviors

1. **Prompt-only (LLM-style)**

```python
msgs = build_messages(prompt="Explain routers in one sentence.")
```

Produces:

```python
[
    {
        "role": "user",
        "content": [{"type": "text", "text": "Explain routers in one sentence."}],
    }
]
```

2. **Prompt + image(s) (VLM-style)**

```python
msgs = build_messages(
    prompt="What is in this image?",
    images="cat.png",
)
```

Produces a multimodal user message with text + image parts.

3. **Manual content parts**

```python
content_parts = [
    {"type": "text", "text": "Question:"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
]

msgs = build_messages(content_parts=content_parts)
```

4. **Optional system message**

```python
msgs = build_messages(
    system="You are a helpful assistant.",
    prompt="Hello!",
)
```

Prepends a system message and then a user message.

---

## 3. Model Helpers (`models.py`)

### Purpose

Light abstraction layer over multiple `ModelEndpoint` instances.

### Key Elements

```python
from which_vlm.inference_api_call.models import (
    ModelEndpoint,      # re-export from config.py
    ModelRegistry,
    build_model_registry,
    filter_models_by_name,
)

ModelRegistry = Dict[str, ModelEndpoint]
```

#### `build_model_registry(models)`

```python
registry = build_model_registry(models)
# registry["llama-8b"] -> ModelEndpoint(...)
```

Used by the runner as `self.models`.

#### `filter_models_by_name(models, names)`

```python
subset = filter_models_by_name(models, ["llama-8b"])
```

Simple helper to get a subset by name.

---

## 4. Runner Layer (`runners.py`)

### Purpose

Wrap the OpenAI Python SDK to:

- Hold a registry of models.
- Create one client per model.
- Provide:
  - Single-model `chat(...)`
  - Multi-model `fanout(...)`
  - Convenience `run_all(...)` for quick multimodal tests.

### Class: `OpenAIStyleRunner`

#### Construction

```python
from which_vlm.inference_api_call.runners import OpenAIStyleRunner

# From loaded models list:
runner = OpenAIStyleRunner(models)

# Direct from YAML/JSON:
runner = OpenAIStyleRunner.from_yaml("models.yaml")
# or
runner = OpenAIStyleRunner.from_json("models.json")
```

#### Attributes

- `runner.models_list` → list of `ModelEndpoint`
- `runner.models` → `ModelRegistry` (name → ModelEndpoint)
- `runner.clients` → dict of `name → OpenAI` client (configured with `base_url` + `api_key`)
- `runner.request_timeout_s`
- `runner.max_workers` (for thread pool fan-out)

#### `chat(model_name, messages, **kwargs)`

```python
msgs = build_messages(prompt="Say hello.")
out = runner.chat("llama-8b", msgs, temperature=0.3)
```

Returns a dict:

```python
{
  "ok": True,
  "model": "llama-8b",
  "model_id": "meta-llama-3-8b-instruct",
  "response_text": "...",
  "raw": <raw response or dict>,
  "latency_ms": 123,
  "usage": {...},          # if server returns usage
  "est_cost": 0.0,
  "conf_proxy": None,
  "request": {...},        # payload sent
}
```

It merges:

- Per-call `kwargs` (like `temperature=`, `max_tokens=`) and
- Model-specific `extra_params`.

> ⚠ Make sure `extra_params` does **not** contain `temperature`, `top_p`, or `max_tokens` if you also pass them in manually, to avoid duplicate keys.

#### `fanout(messages, model_names=None, **kwargs)`

Call multiple models in parallel:

```python
msgs = build_messages(prompt="Explain routers in one sentence.")
results = runner.fanout(messages=msgs, model_names=["llama-8b", "qwen3-8b"])

for name, out in results.items():
    print(name, "->", out.get("response_text"))
```

If `model_names` is `None`, all configured models are used.

#### `run_all(prompt=None, images=None, content_parts=None, system=None, model_names=None, **kwargs)`

Shortcut: build messages via `build_messages` and fan out.

```python
results = runner.run_all(
    prompt="What is happening in this image?",
    images="example.jpg",
)

for name, out in results.items():
    print(name, "->", out["response_text"])
```

---

## 5. Test Suites (`suites.py`)

### Purpose

Provide **ergonomic, experiment-focused wrappers** on top of the runner:

- `LLMTestSuite` – for text-only LLM experiments.
- `VLMTestSuite` – for image(+text) VLM experiments.

They know **which models they’re allowed to use** and rely on the runner to do actual calls.

### Base: `BaseSuite`

```python
@dataclass
class BaseSuite:
    runner: OpenAIStyleRunner
    model_names: List[str]

    def _resolve_models(self, models=None) -> List[str]:
        ...

    def list_models(self) -> List[str]:
        ...
```

- `_resolve_models(models)`:
  - `None` or `"all"` → all suite models
  - `"llama-8b"` → `["llama-8b"]`
  - `["llama-8b", "qwen3-8b"]` → validated subset
- `list_models()` → returns `model_names`.

---

### `LLMTestSuite` – Text-only LLMs

#### Construction

```python
from which_vlm.inference_api_call.suites import LLMTestSuite

llm_names = ["llama-8b", "gpt4-mini"]
llm = LLMTestSuite(runner, llm_names)
```

#### `run_single(prompt, models="all", system=None, **gen_kwargs)`

```python
prompt = "Explain what a router is in one simple sentence."
res = llm.run_single(prompt, models="all")

for name, out in res.items():
    print("LLM:", name, "->", out["response_text"])
```

- Uses `build_messages(prompt=..., system=...)`.
- Calls `runner.fanout(...)` under the hood.

#### `run_batch(prompts, model, system=None, **gen_kwargs)`

```python
prompts = ["1+1=", "2+2=", "What is a router?"]
results = llm.run_batch(prompts, model="llama-8b")

for p, out in zip(prompts, results):
    print(p, "->", out["response_text"])
```

- Only supports a **single model name**.
- Calls `runner.chat(...)` once per prompt.

#### `compare_models(prompt, models="all", system=None, **gen_kwargs)`

```python
comparisons = llm.compare_models("Give me a one-line joke about routers.")
for name, text in comparisons.items():
    print(name, "->", text)
```

- Wraps `run_single` and returns only `response_text` per model.

---

### `VLMTestSuite` – Image(+text) VLMs

#### Construction

```python
from which_vlm.inference_api_call.suites import VLMTestSuite

vlm_names = ["qwen3-8b-vl"]
vlm = VLMTestSuite(runner, vlm_names)
```

#### `run_image(image, text=None, models="all", system=None, **gen_kwargs)`

```python
vlm_results = vlm.run_image(
    image="example.jpg",
    text="What is happening in this image?",
    models="all",
)

for name, out in vlm_results.items():
    print("VLM:", name, "->", out["response_text"])
```

- `image` can be any `ImageLike` (path, URL, bytes, PIL, numpy).
- Uses `build_messages(prompt=text, images=image, system=system)`.

#### `run_batch(examples, models="all", system=None, **gen_kwargs)`

```python
examples = [
    {"image": "cat.png", "text": "What animal is this?"},
    {"image": "dog.png", "text": "What animal is this?"},
]

vlm_batch = vlm.run_batch(examples, models="all")

for model_name, outputs in vlm_batch.items():
    print("Model:", model_name)
    for i, out in enumerate(outputs):
        print(f"  Example {i} ->", out["response_text"])
```

- `examples` is an iterable of dicts with keys:
  - `"image"`: `ImageLike`
  - `"text"`: `Optional[str]`

#### `describe_image(image, models="all", system=None, **gen_kwargs)`

```python
desc = vlm.describe_image("example.jpg", models="all")
for name, text in desc.items():
    print(name, "->", text)
```

- Shortcut for “Descrv
- Returns `model_name -> response_text`.

---

## 6. High-Level Client (`client.py`)

### Purpose

`WhichVLMClient` is the main **user-facing entrypoint** that ties together:

- Config loading
- Runner
- LLM/VLM suites

You use this in notebooks and scripts.

### Class: `WhichVLMClient`

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import ModelEndpoint, load_models_from_yaml, load_models_from_json, split_models_by_type
from .runners import OpenAIStyleRunner
from .suites import LLMTestSuite, VLMTestSuite

@dataclass
class WhichVLMClient:
    runner: OpenAIStyleRunner
    llm: LLMTestSuite
    vlm: VLMTestSuite
```

#### Constructing the client

**From YAML:**

```python
from which_vlm.inference_api_call.client import WhichVLMClient

client = WhichVLMClient.from_yaml(
    "models.yaml",
    request_timeout_s=120,
    max_workers=4,
)
```

**From JSON:**

```python
client = WhichVLMClient.from_json("models.json")
```

**From in-memory models:**

```python
from which_vlm.inference_api_call.config import load_models_from_yaml

models = load_models_from_yaml("models.yaml")
client = WhichVLMClient.from_models(models)
```

If `llm_names` / `vlm_names` are not provided, `from_models` uses `split_models_by_type(models)` and the `model_type` field to build the suites.

#### Introspection helpers

```python
client.list_models()      # All models known to the runner
client.list_llm_models()  # Models wired into LLM suite
client.list_vlm_models()  # Models wired into VLM suite
client.split_models()     # (llm_names, vlm_names)
```

#### Using `client.llm` and `client.vlm`

**LLM example:**

```python
prompt = "Explain what a router is in one simple sentence."

llm_results = client.llm.run_single(prompt, models="all")

for name, out in llm_results.items():
    print("=== LLM:", name, "===")
    if out.get("ok"):
        print(out["response_text"])
        print("Latency (ms):", out["latency_ms"])
        print("Estimated cost:", out["est_cost"])
    else:
        print("ERROR:", out.get("error"))
```

**VLM example:**

```python
image_path = "example.jpg"

vlm_results = client.vlm.run_image(
    image=image_path,
    text="What is happening in this image?",
    models="all",
)

for name, out in vlm_results.items():
    print("=== VLM:", name, "===")
    if out.get("ok"):
        print(out["response_text"])
        print("Latency (ms):", out["latency_ms"])
    else:
        print("ERROR:", out.get("error"))
```

---

## 7. Example `models.yaml` Config

Here is a concrete template you can adapt:

```yaml
models:
  - name: llama-8b
    base_url: http://10.137.22.160:9000/v1
    api_key: EMPTY
    model_id: meta-llama-3-8b-instruct
    model_type: llm
    pricing:
      prompt_per_1k: 0.0
      completion_per_1k: 0.0
    extra_params: {}   # IMPORTANT: don't duplicate `temperature` here

  - name: qwen3-vl-8b
    base_url: http://10.137.22.160:9000/v1
    api_key: EMPTY
    model_id: qwen/qwen3-vl-8b
    model_type: vlm
    pricing:
      prompt_per_1k: 0.0
      completion_per_1k: 0.0
    extra_params: {}
```

Once you have this file, you can spin up a client in Jupyter:

```python
from which_vlm.inference_api_call.client import WhichVLMClient

client = WhichVLMClient.from_yaml("models.yaml")

print("All models:", client.list_models())
print("LLM models:", client.list_llm_models())
print("VLM models:", client.list_vlm_models())
```

---

## 8. Troubleshooting Tips

### 1. `KeyError: "Unknown model: ..."`

- You passed a model name that is not in the config.
- Check `client.list_models()` to see the available names.
- Confirm `name:` in YAML matches exactly.

### 2. `dict() got multiple values for keyword argument 'temperature'`

- You likely defined `temperature` in `extra_params` **and** passed `temperature=` in code (or used defaults).
- Fix: remove `temperature` from `extra_params` and control it only via `run_single(..., temperature=...)`.

### 3. Image errors (file not found / unsupported type)

- Double-check paths: they are relative to the notebook’s working directory.
- Ensure `PIL` and `numpy` are installed if you pass PIL images or numpy arrays.
- For URLs, ensure your model server supports remote image URLs (some expect data URLs only).

### 4. Connection / HTTP errors

- Verify your `base_url` is reachable.
- Check that the server is running and exposing an OpenAI-compatible `/v1/chat/completions` endpoint.
- If using vLLM or LM Studio, test with a minimal curl / Python OpenAI call first.

---

## 9. Recommended Usage Patterns

- Use `WhichVLMClient` as your **main entrypoint**; rarely interact directly with `OpenAIStyleRunner` unless you need custom behavior.
- Use `client.llm` for any **pure text** experiments.
- Use `client.vlm` for any **image + text** tests.
- Keep `models.yaml` as your single source of truth for the endpoints; version control it alongside your experiments.

This README should serve as your **reference map** for what each file, class, and function does, and how to use them in practice.