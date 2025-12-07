

"""
Model-layer abstractions for Which_VLM Router.

This module provides small, shared types and helpers that sit between the
raw config (YAML / JSON) and the higher-level runner / test suites.

Right now it primarily re-exports the `ModelEndpoint` dataclass defined in
`config.py`, plus a few convenience types / helpers for working with
collections of models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .config import ModelEndpoint  # single source of truth for the dataclass

__all__ = [
    "ModelEndpoint",
    "ModelRegistry",
    "build_model_registry",
    "filter_models_by_name",
]


# A registry is the common structure used by runners: name -> ModelEndpoint
ModelRegistry = Dict[str, ModelEndpoint]


def build_model_registry(models: Iterable[ModelEndpoint]) -> ModelRegistry:
    """
    Build a simple name -> ModelEndpoint mapping from an iterable.

    Later, the runner will typically store this mapping as `self.models`.
    """
    return {m.name: m for m in models}


def filter_models_by_name(
    models: Iterable[ModelEndpoint],
    names: List[str],
) -> List[ModelEndpoint]:
    """
    Filter a list / iterable of ModelEndpoint objects by a list of names.

    This is a convenience helper that can be useful when constructing
    LLM / VLM suites from a shared config.
    """
    name_set = set(names)
    return [m for m in models if m.name in name_set]