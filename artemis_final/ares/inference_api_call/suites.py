

"""
Test suites for LLM and VLM models.

These classes provide a thin, ergonomic layer on top of the generic
OpenAIStyleRunner so that you can:

- Run text-only prompts across one / many LLMs.
- Run image(+text) prompts across one / many VLMs.
- Do quick batch experiments without worrying about message wiring.

They are deliberately simple and notebook-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .messages import ImageLike, build_messages
from .runners import OpenAIStyleRunner


# ---------------------------------------------------------------------------
# Base suite
# ---------------------------------------------------------------------------


@dataclass
class BaseSuite:
    """
    Shared base class for LLMTestSuite and VLMTestSuite.

    It primarily encapsulates:
    - a reference to the underlying OpenAIStyleRunner
    - the subset of model names this suite is allowed to use
    """

    runner: OpenAIStyleRunner
    model_names: List[str]

    def _resolve_models(
        self,
        models: Optional[Union[str, Sequence[str]]] = None,
    ) -> List[str]:
        """
        Resolve the `models` argument into a concrete list of model names.

        Rules:
        - None or "all" -> all model_names known to this suite.
        - str           -> single-element list [str].
        - sequence      -> list(models), but validated to exist in runner.
        """
        if models is None or models == "all":
            resolved = list(self.model_names)
        elif isinstance(models, str):
            resolved = [models]
        else:
            resolved = list(models)

        # Validate
        unknown = [m for m in resolved if m not in self.runner.models]
        if unknown:
            raise KeyError(f"Unknown model(s) for this suite: {unknown}")

        return resolved

    # Simple convenience for introspection
    def list_models(self) -> List[str]:
        """Return the list of model names this suite is configured to use."""
        return list(self.model_names)


# ---------------------------------------------------------------------------
# LLM suite (text-only)
# ---------------------------------------------------------------------------


class LLMTestSuite(BaseSuite):
    """
    Helpers for running text-only experiments on LLM models.

    Typical usage from a notebook:

    >>> llm = LLMTestSuite(runner, llm_model_names)
    >>> llm.run_single("Explain routers in one sentence.", models="all")
    """

    def run_single(
        self,
        prompt: str,
        models: Optional[Union[str, Sequence[str]]] = "all",
        system: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a single text prompt on one or multiple LLM models.

        Parameters
        ----------
        prompt:
            User text prompt.
        models:
            - "all" / None -> use all LLM models in this suite.
            - str          -> single model name.
            - sequence     -> subset of model names.
        system:
            Optional system prompt.
        **gen_kwargs:
            Extra generation parameters forwarded to the underlying runner.

        Returns
        -------
        dict
            Mapping: model_name -> result dict from runner.chat.
        """
        resolved = self._resolve_models(models)
        msgs = build_messages(prompt=prompt, system=system)
        return self.runner.fanout(messages=msgs, model_names=resolved, **gen_kwargs)

    def run_batch(
        self,
        prompts: Iterable[str],
        model: str,
        system: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Run a list of prompts on a single LLM model.

        This is useful for quick, per-model evaluation or logging.

        Parameters
        ----------
        prompts:
            Iterable of user prompts.
        model:
            Single LLM model name to query.
        system:
            Optional system prompt.
        **gen_kwargs:
            Extra generation parameters forwarded to the underlying runner.

        Returns
        -------
        list
            List of result dicts in the same order as the input prompts.
        """
        resolved = self._resolve_models(model)
        if len(resolved) != 1:
            raise ValueError("run_batch expects a single model name, not multiple.")
        model_name = resolved[0]

        results: List[Dict[str, Any]] = []
        for p in prompts:
            msgs = build_messages(prompt=p, system=system)
            out = self.runner.chat(model_name, messages=msgs, **gen_kwargs)
            results.append(out)
        return results

    def compare_models(
        self,
        prompt: str,
        models: Optional[Union[str, Sequence[str]]] = "all",
        system: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> Dict[str, str]:
        """
        Convenience wrapper that returns just the response_text per model.

        Parameters
        ----------
        prompt:
            User text prompt.
        models:
            "all" / None / str / list of str.
        system:
            Optional system prompt.
        **gen_kwargs:
            Extra generation parameters forwarded to the underlying runner.

        Returns
        -------
        dict
            Mapping: model_name -> response_text.
        """
        full = self.run_single(prompt=prompt, models=models, system=system, **gen_kwargs)
        return {name: (res.get("response_text") or "") for name, res in full.items()}


# ---------------------------------------------------------------------------
# VLM suite (image + text)
# ---------------------------------------------------------------------------


class VLMTestSuite(BaseSuite):
    """
    Helpers for running image(+text) experiments on VLM models.

    Typical usage from a notebook:

    >>> vlm = VLMTestSuite(runner, vlm_model_names)
    >>> vlm.run_image("cat.png", "What is the cat doing?", models="all")
    """

    def run_image(
        self,
        image: ImageLike,
        text: Optional[str] = None,
        models: Optional[Union[str, Sequence[str]]] = "all",
        system: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a single (image, optional text) pair on one or multiple VLM models.

        Parameters
        ----------
        image:
            Any supported ImageLike input:
            - file path (str)
            - HTTP/HTTPS URL (str)
            - data URL or bare base64 string
            - bytes / bytearray
            - PIL.Image.Image
            - numpy.ndarray
        text:
            Optional user text to accompany the image (e.g. a question).
        models:
            "all" / None / str / list of str.
        system:
            Optional system prompt.
        **gen_kwargs:
            Extra generation parameters forwarded to the underlying runner.

        Returns
        -------
        dict
            Mapping: model_name -> result dict from runner.chat.
        """
        resolved = self._resolve_models(models)
        msgs = build_messages(prompt=text, images=image, system=system)
        return self.runner.fanout(messages=msgs, model_names=resolved, **gen_kwargs)

    def run_batch(
        self,
        examples: Iterable[Dict[str, Any]],
        models: Optional[Union[str, Sequence[str]]] = "all",
        system: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run a batch of (image, text) examples across one or multiple VLM models.

        Each example is expected to be a dict with at least:
            {"image": <ImageLike>, "text": <Optional[str]>}

        Parameters
        ----------
        examples:
            Iterable of example dicts as described above.
        models:
            "all" / None / str / list of str.
        system:
            Optional system prompt.
        **gen_kwargs:
            Extra generation parameters forwarded to the underlying runner.

        Returns
        -------
        dict
            Mapping: model_name -> list of result dicts (one per example, in order).
        """
        resolved = self._resolve_models(models)
        examples_list = list(examples)

        results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in resolved}
        for model_name in resolved:
            for ex in examples_list:
                if "image" not in ex:
                    raise ValueError("Each example must contain an 'image' key.")
                img = ex["image"]
                text = ex.get("text")
                msgs = build_messages(prompt=text, images=img, system=system)
                out = self.runner.chat(model_name, messages=msgs, **gen_kwargs)
                results[model_name].append(out)

        return results

    def describe_image(
        self,
        image: ImageLike,
        models: Optional[Union[str, Sequence[str]]] = "all",
        system: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> Dict[str, str]:
        """
        Shortcut for "describe this image" across multiple VLM models.

        Parameters
        ----------
        image:
            Any supported ImageLike input.
        models:
            "all" / None / str / list of str.
        system:
            Optional system prompt.
        **gen_kwargs:
            Extra generation parameters forwarded to the underlying runner.

        Returns
        -------
        dict
            Mapping: model_name -> response_text.
        """
        full = self.run_image(
            image=image,
            text="Describe this image.",
            models=models,
            system=system,
            **gen_kwargs,
        )
        return {name: (res.get("response_text") or "") for name, res in full.items()}