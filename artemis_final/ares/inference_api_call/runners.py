

"""
Runners: thin wrappers around OpenAI-style endpoints (LLM + VLM).

This module provides a generic runner that:
- Holds a registry of ModelEndpoint objects.
- Creates one OpenAI client per model (base_url + api_key).
- Exposes simple methods for:
    * single-model chat calls
    * parallel fan-out to multiple models
    * convenience multimodal calls via `build_messages`

It is intentionally synchronous (thread-based fan-out) and avoids any
framework-specific dependencies so it can be dropped into notebooks or
small scripts easily.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import load_models_from_json, load_models_from_yaml, ModelEndpoint
from .messages import build_messages
from .models import ModelRegistry, build_model_registry


class OpenAIStyleRunner:
    """
    Generic OpenAI-style chat runner for both LLM and VLM models.

    Parameters
    ----------
    models:
        Iterable of ModelEndpoint objects describing all available models.
    request_timeout_s:
        Timeout in seconds for each individual request (passed to the client).
    max_workers:
        Maximum number of threads to use for parallel fan-out.
    """

    def __init__(
        self,
        models: List[ModelEndpoint],
        request_timeout_s: int = 120,
        max_workers: int = 4,
    ) -> None:
        self.models_list: List[ModelEndpoint] = models
        self.models: ModelRegistry = build_model_registry(models)

        # One OpenAI client per model.
        # This keeps the logic simple and avoids having to group by base_url.
        self.clients: Dict[str, OpenAI] = {
            m.name: OpenAI(api_key=m.api_key, base_url=m.base_url)
            for m in models
        }

        self.request_timeout_s = request_timeout_s
        self.max_workers = max_workers

    # ---------------------------------------------------------------------
    # Alternative constructors
    # ---------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: str,
        request_timeout_s: int = 120,
        max_workers: int = 4,
    ) -> "OpenAIStyleRunner":
        """
        Build a runner from a YAML config file (see config.load_models_from_yaml).
        """
        models = load_models_from_yaml(path)
        return cls(models=models, request_timeout_s=request_timeout_s, max_workers=max_workers)

    @classmethod
    def from_json(
        cls,
        path: str,
        request_timeout_s: int = 120,
        max_workers: int = 4,
    ) -> "OpenAIStyleRunner":
        """
        Build a runner from a JSON config file (see config.load_models_from_json).
        """
        models = load_models_from_json(path)
        return cls(models=models, request_timeout_s=request_timeout_s, max_workers=max_workers)

    # ---------------------------------------------------------------------
    # Core single-model call
    # ---------------------------------------------------------------------

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send a single OpenAI Chat Completions request to one model.

        Parameters
        ----------
        model_name:
            Logical name of the model as defined in the config (ModelEndpoint.name).
        messages:
            OpenAI-style messages payload (e.g. from `build_messages`).
        **kwargs:
            Generation kwargs such as `temperature`, `max_tokens`, etc.
            These are merged with the model's `extra_params`.

        Returns
        -------
        dict
            A dictionary with keys:
            - ok: bool
            - model: str
            - model_id: str
            - response_text: str
            - raw: raw response (dict or pydantic model)
            - latency_ms: int
            - usage: dict (if available)
            - est_cost: float
            - conf_proxy: Optional[float] (always None here, reserved for future)
            - request: the payload used for the call
        """
        if model_name not in self.models:
            raise KeyError(f"Unknown model: {model_name}")

        endpoint = self.models[model_name]
        client = self.clients[model_name]
        temp = kwargs.pop(
            "temperature",
            endpoint.default_temperature,  # <-- Use model default if no manual override
        )
        # Default parameters with per-model extras.
        payload: Dict[str, Any] = dict(
            model=endpoint.model_id,
            messages=messages,
            temperature=temp,
            top_p=kwargs.pop("top_p", 1.0),
            max_tokens=kwargs.pop("max_tokens", 512),
            logprobs=kwargs.pop("logprobs", True),  # Enable logprobs by default
            top_logprobs=kwargs.pop("top_logprobs", 1),  # Get top 1 logprobs
            **endpoint.extra_params,
            **kwargs,
        )

        t0 = time.perf_counter()
        resp = client.chat.completions.create(**payload)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        choice = resp.choices[0]
        response_text = getattr(choice.message, "content", "") or ""
        
        # Extract logprobs from choice
        choice_logprobs = getattr(choice, "logprobs", None)
        logprobs_data = None
        if choice_logprobs is not None:
            if hasattr(choice_logprobs, "model_dump"):
                logprobs_data = choice_logprobs.model_dump()
            elif hasattr(choice_logprobs, "content"):
                logprobs_data = {"content": choice_logprobs.content}
            else:
                logprobs_data = choice_logprobs

        usage = getattr(resp, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
        est_cost = self._estimate_cost(usage, endpoint.pricing)

        return {
            "ok": True,
            "model": model_name,
            "model_id": endpoint.model_id,
            "response_text": response_text,
            "raw": resp.model_dump() if hasattr(resp, "model_dump") else resp,
            "logprobs": logprobs_data,  # NEW: Return logprobs
            "latency_ms": latency_ms,
            "usage": usage_dict,
            "input_tokens": usage_dict.get("prompt_tokens", 0),  # NEW: Extract input tokens
            "output_tokens": usage_dict.get("completion_tokens", 0),  # NEW: Extract output tokens
            "est_cost": est_cost,
            "conf_proxy": None,
            "request": payload,
        }

    # ---------------------------------------------------------------------
    # Fan-out to multiple models (threaded)
    # ---------------------------------------------------------------------

    def fanout(
        self,
        messages: List[Dict[str, Any]],
        model_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Call multiple models in parallel using ThreadPoolExecutor.

        Parameters
        ----------
        messages:
            OpenAI-style messages payload.
        model_names:
            Optional list of model names. If None, calls *all* known models.
        **kwargs:
            Extra generation parameters forwarded to `chat`.

        Returns
        -------
        dict
            Mapping: model_name -> result dict (or error dict with ok=False).
        """
        if model_names is None:
            model_names = list(self.models.keys())

        results: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.chat, name, messages, **kwargs): name
                for name in model_names
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results[name] = fut.result()
                except Exception as exc:  # pragma: no cover - defensive
                    results[name] = {
                        "ok": False,
                        "model": name,
                        "error": str(exc),
                    }
        return results

    # ---------------------------------------------------------------------
    # Convenience wrapper for building messages + fan-out
    # ---------------------------------------------------------------------

    def run_all(
        self,
        prompt: Optional[str] = None,
        images: Optional[Any] = None,
        content_parts: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convenience wrapper that:
        - builds messages via `build_messages`
        - fans out the request to multiple models.

        This is handy for quick multimodal experiments in notebooks.

        Parameters
        ----------
        prompt:
            Optional user text prompt.
        images:
            Optional image or list of images (see `build_messages` for supported types).
        content_parts:
            Optional pre-built content parts (overrides prompt + images).
        system:
            Optional system prompt.
        model_names:
            Optional list of models to call. If None, all models are used.
        **kwargs:
            Extra generation parameters forwarded to `chat`.

        Returns
        -------
        dict
            Mapping: model_name -> result dict.
        """
        msgs = build_messages(
            prompt=prompt,
            images=images,
            content_parts=content_parts,
            system=system,
        )
        return self.fanout(messages=msgs, model_names=model_names, **kwargs)

    # ---------------------------------------------------------------------
    # Cost helper
    # ---------------------------------------------------------------------

    @staticmethod
    def _estimate_cost(usage: Any, pricing: Dict[str, float]) -> float:
        """
        Rough cost estimation given OpenAI-style usage and a pricing dict.

        Expected keys in `pricing`:
        - prompt_per_1k
        - completion_per_1k

        If usage or pricing is missing, returns 0.0.
        """
        if not usage or not pricing:
            return 0.0

        try:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or usage.get("prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0) or usage.get(
                "completion_tokens", 0
            )
        except Exception:  # pragma: no cover - very defensive
            prompt_tokens = 0
            completion_tokens = 0

        return (
            (prompt_tokens / 1000.0) * pricing.get("prompt_per_1k", 0.0)
            + (completion_tokens / 1000.0) * pricing.get("completion_per_1k", 0.0)
        )