

"""
High-level client entrypoint for Which_VLM Router.

This module ties together:

- Model configuration loading (from YAML / JSON)
- The generic OpenAIStyleRunner
- LLM and VLM test suites

so that, from a notebook or script, you can do:

    from which_vlm.inference_api_call.client import WhichVLMClient

    client = WhichVLMClient.from_yaml("models.yaml")

    client.list_models()
    client.llm.run_single("Explain routers in one sentence.", models="all")
    client.vlm.run_image("cat.png", "What is the cat doing?", models="all")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import (
    ModelEndpoint,
    load_models_from_json,
    load_models_from_yaml,
    split_models_by_type,
)
from .runners import OpenAIStyleRunner
from .suites import LLMTestSuite, VLMTestSuite


@dataclass
class WhichVLMClient:
    """
    High-level convenience client that exposes:

    - `llm`: an LLMTestSuite for text-only experiments.
    - `vlm`: a VLMTestSuite for image(+text) experiments.
    - simple helpers to inspect available models.

    Instances are typically constructed via `from_yaml` or `from_json`,
    which load model configurations and wire everything up.
    """

    runner: OpenAIStyleRunner
    llm: LLMTestSuite
    vlm: VLMTestSuite

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_models(
        cls,
        models: List[ModelEndpoint],
        request_timeout_s: int = 300,
        max_workers: int = 4,
        llm_names: Optional[List[str]] = None,
        vlm_names: Optional[List[str]] = None,
    ) -> "WhichVLMClient":
        """
        Build a WhichVLMClient directly from a list of ModelEndpoint objects.

        If `llm_names` and/or `vlm_names` are not provided, they are derived
        from the `model_type` field via `split_models_by_type`.
        """
        # If caller didn't specify explicit splits, derive them.
        if llm_names is None or vlm_names is None:
            auto_llm_names, auto_vlm_names = split_models_by_type(models)
            llm_names = llm_names or auto_llm_names
            vlm_names = vlm_names or auto_vlm_names

        # Initialise the shared runner
        runner = OpenAIStyleRunner(
            models=models,
            request_timeout_s=request_timeout_s,
            max_workers=max_workers,
        )

        # Create suites
        llm_suite = LLMTestSuite(runner=runner, model_names=llm_names)
        vlm_suite = VLMTestSuite(runner=runner, model_names=vlm_names)

        return cls(runner=runner, llm=llm_suite, vlm=vlm_suite)

    @classmethod
    def from_yaml(
        cls,
        path: str,
        request_timeout_s: int = 300,
        max_workers: int = 4,
    ) -> "WhichVLMClient":
        """
        Build a WhichVLMClient from a YAML config file.

        See `config.load_models_from_yaml` and the docstring there for
        details on the expected file format.
        """
        models = load_models_from_yaml(path)
        return cls.from_models(
            models=models,
            request_timeout_s=request_timeout_s,
            max_workers=max_workers,
        )

    @classmethod
    def from_json(
        cls,
        path: str,
        request_timeout_s: int = 300,
        max_workers: int = 4,
    ) -> "WhichVLMClient":
        """
        Build a WhichVLMClient from a JSON config file.

        See `config.load_models_from_json` and the docstring there for
        details on the expected file format.
        """
        models = load_models_from_json(path)
        return cls.from_models(
            models=models,
            request_timeout_s=request_timeout_s,
            max_workers=max_workers,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return all model names known to the underlying runner."""
        return list(self.runner.models.keys())

    def list_llm_models(self) -> List[str]:
        """Return the model names configured for the LLM test suite."""
        return self.llm.list_models()

    def list_vlm_models(self) -> List[str]:
        """Return the model names configured for the VLM test suite."""
        return self.vlm.list_models()

    def split_models(self) -> Tuple[List[str], List[str]]:
        """
        Return (llm_names, vlm_names) as seen by this client.

        This mirrors the output of `split_models_by_type` but uses
        the currently configured suites.
        """
        return self.list_llm_models(), self.list_vlm_models()