

"""
Configuration utilities for Which_VLM Router.

This module is responsible for:
- Defining the ModelEndpoint dataclass (one instance per hosted model).
- Loading model configurations from YAML / JSON files.
- Providing small helpers to split models into LLM / VLM groups.

The config file format is intentionally simple. A typical YAML looks like:

models:
  - name: qwen3-8b
    base_url: http://localhost:8001/v1
    api_key: EMPTY
    model_id: Qwen/Qwen3-8B-Instruct
    model_type: llm          # "llm" or "vlm" (default: "llm")
    pricing:
      prompt_per_1k: 0.0
      completion_per_1k: 0.0
    extra_params:
      temperature: 0.2

  - name: qwen2p5-vl-7b
    base_url: http://localhost:8002/v1
    api_key: EMPTY
    model_id: Qwen/Qwen2.5-VL-7B-Instruct
    model_type: vlm
    pricing: {}
    extra_params: {}

You can also use JSON with the exact same structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelEndpoint:
    """
    Represents a single OpenAI-compatible model endpoint.

    Attributes
    ----------
    name:
        Short logical name you use in code (e.g. "qwen3-8b").
    base_url:
        Base URL for the OpenAI-style endpoint (e.g. "http://localhost:8001/v1").
    api_key:
        API key/token required by the server. For many local deployments this can
        be "EMPTY" or any non-empty string.
    model_id:
        The model identifier passed to the OpenAI client as the `model` field
        (e.g. "Qwen/Qwen3-8B-Instruct").
    pricing:
        Pricing config, typically:
        {
            "prompt_per_1k": float,
            "completion_per_1k": float
        }
        but any keys are accepted.
    extra_params:
        Extra parameters that should always be sent for this model when making
        chat requests (e.g. {"temperature": 0.2}).
    model_type:
        Logical type of the model: "llm", "vlm", or "both".
        This is used only by higher-level routing / test suites.
    """

    name: str
    base_url: str
    api_key: str
    model_id: str
    pricing: Dict[str, float]
    extra_params: Dict[str, Any]
    model_type: str = "vlm"  # "llm", "vlm", or "both"
    default_temperature: float = 0.0


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _normalise_path(path: str | Path) -> Path:
    """Return a resolved Path for convenience."""
    return Path(path).expanduser().resolve()


def _parse_models_from_dict(cfg: Dict[str, Any]) -> List[ModelEndpoint]:
    """
    Internal helper to turn a raw config dict into a list of ModelEndpoint objects.

    Expected top-level structure:
        {"models": [ {...}, {...}, ... ]}
    """
    if "models" not in cfg or not isinstance(cfg["models"], Iterable):
        raise ValueError("Config must contain a top-level 'models' list.")

    models: List[ModelEndpoint] = []
    for raw in cfg["models"]:
        if not isinstance(raw, dict):
            raise ValueError("Each entry in 'models' must be a dict.")

        name = raw["name"]
        base_url = str(raw["base_url"]).rstrip("/")
        api_key = raw.get("api_key", "EMPTY") or "EMPTY"
        model_id = raw["model_id"]
        pricing = raw.get("pricing", {}) or {}
        extra_params = raw.get("extra_params", {}) or {}
        model_type = raw.get("model_type", "llm")

        models.append(
            ModelEndpoint(
                name=name,
                base_url=base_url,
                api_key=api_key,
                model_id=model_id,
                pricing=pricing,
                extra_params=extra_params,
                model_type=model_type,
            )
        )

    return models


def load_models_from_yaml(path: str | Path) -> List[ModelEndpoint]:
    """
    Load a list of ModelEndpoint definitions from a YAML file.

    Parameters
    ----------
    path:
        Path to a YAML file with a top-level `models` list.

    Returns
    -------
    List[ModelEndpoint]
    """
    p = _normalise_path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _parse_models_from_dict(cfg)


def load_models_from_json(path: str | Path) -> List[ModelEndpoint]:
    """
    Load a list of ModelEndpoint definitions from a JSON file.

    Parameters
    ----------
    path:
        Path to a JSON file with a top-level `models` list.

    Returns
    -------
    List[ModelEndpoint]
    """
    p = _normalise_path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f) or {}
    return _parse_models_from_dict(cfg)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def split_models_by_type(
    models: List[ModelEndpoint],
) -> Tuple[List[str], List[str]]:
    """
    Split model names into (llm_names, vlm_names) based on `model_type`.

    Any model with `model_type == "vlm"` is treated as VLM-only.
    Any model with `model_type == "llm"` is treated as LLM-only.
    Any model with `model_type == "both"` is included in both lists.

    Returns
    -------
    (llm_names, vlm_names)
    """
    llm_names: List[str] = []
    vlm_names: List[str] = []

    for m in models:
        t = (m.model_type or "llm").lower()
        if t in ("llm", "both"):
            llm_names.append(m.name)
        if t in ("vlm", "both"):
            vlm_names.append(m.name)

    return llm_names, vlm_names