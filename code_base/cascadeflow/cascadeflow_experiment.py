"""
High-level helpers for running CascadeFlow VLM routing experiments.

This module packages the notebook logic so that anything that requires
`async`/`await` lives in importable helper functions. The notebook (or any
script) can simply build an `ExperimentConfig` instance and call
`run_experiment(...)` without worrying about event loops.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import sys
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments

    def tqdm(iterable, **kwargs):  # type: ignore
        """Fallback tqdm replacement when the real dependency is unavailable."""

        return iterable


from cascadeflow import CascadeAgent, ModelConfig


@dataclass
class ModelSpec:
    """Declarative description of a model slot in the cascade."""

    name: str
    base_url: str
    cost: float
    provider: str = "vllm"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    quality_score: Optional[float] = None
    speed_ms: Optional[int] = None
    keywords: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """
    Configuration bundle for CascadeFlow routing experiments.

    Attributes mirror the notebook knobs so callers can tweak dataset paths,
    sampling, generation params, and output destinations from plain Python.
    """

    dataset_path: Path
    cauldron_lookup_path: Path
    image_root: Path
    output_dir: Path
    cascade_models: list[ModelSpec]
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    max_samples: Optional[int] = 100
    seed: int = 42
    generation_max_tokens: int = 300
    generation_temperature: float = 0.1
    verbose_agent: bool = False
    experiment_name: str = "cascadeflow_vlm_router"

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path).expanduser().resolve()
        self.cauldron_lookup_path = Path(self.cauldron_lookup_path).expanduser().resolve()
        self.image_root = Path(self.image_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WHICH_VLM_SRC = PROJECT_ROOT / "code_base" / "which_vlm"
if WHICH_VLM_SRC.exists():
    which_vlm_str = str(WHICH_VLM_SRC)
    if which_vlm_str not in sys.path:
        sys.path.append(which_vlm_str)

try:
    from dataset_builder.check_data_utils import fetch_cauldron_image
except ImportError:  # pragma: no cover - optional dependency
    fetch_cauldron_image = None

logger = logging.getLogger(__name__)


DEFAULT_MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        name="qwen2.5-vl-3b",
        base_url="http://localhost:7001/v1",
        cost=0.00015,
        temperature=0.1,
        speed_ms=400,
        keywords=["multimodal"],
        domains=["vlm"],
    ),
    ModelSpec(
        name="gemma-3-27b",
        base_url="http://localhost:7002/v1",
        cost=0.0006,
        temperature=0.15,
        speed_ms=900,
        keywords=["multimodal"],
        domains=["vlm"],
    ),
    ModelSpec(
        name="qwen3-vl-8b-thinking",
        base_url="http://localhost:7003/v1",
        cost=0.0015,
        temperature=0.2,
        speed_ms=1500,
        keywords=["multimodal"],
        domains=["vlm"],
    ),
]


def create_default_config(project_root: Path | None = None) -> ExperimentConfig:
    """Convenience helper that mirrors the original notebook defaults."""

    root = project_root or PROJECT_ROOT
    return ExperimentConfig(
        dataset_path=root / "dataset/final_dataset/router_sample_records_test.parquet",
        cauldron_lookup_path=root / "dataset/which_vlm_data/processed/cauldron_poc_multi.parquet",
        image_root=root / "dataset/which_vlm_data/images/cauldron",
        output_dir=root / "dataset/which_vlm_data/results",
        cascade_models=DEFAULT_MODEL_SPECS,
        project_root=root,
    )


def build_model_config(spec: ModelSpec, config: ExperimentConfig) -> ModelConfig:
    """Translate a ModelSpec into the pydantic-backed ModelConfig."""

    kwargs: dict[str, Any] = {
        "name": spec.name,
        "provider": spec.provider,
        "cost": spec.cost,
        "base_url": spec.base_url,
    }

    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    if spec.max_tokens is not None:
        kwargs["max_tokens"] = spec.max_tokens
    if spec.system_prompt is not None:
        kwargs["system_prompt"] = spec.system_prompt
    if spec.quality_score is not None:
        kwargs["quality_score"] = spec.quality_score
    if spec.speed_ms is not None:
        kwargs["speed_ms"] = spec.speed_ms
    if spec.keywords:
        kwargs["keywords"] = spec.keywords
    if spec.domains:
        kwargs["domains"] = spec.domains
    if spec.extra:
        kwargs["extra"] = spec.extra

    return ModelConfig(**kwargs)


def build_cascade_agent(config: ExperimentConfig) -> CascadeAgent:
    """Instantiate CascadeAgent using the provided model specs."""

    models = [build_model_config(spec, config) for spec in config.cascade_models]
    return CascadeAgent(models=models, verbose=config.verbose_agent)


def build_cauldron_lookup(table_path: Path, image_root: Path) -> dict[str, Path]:
    """
    Map `dataset:index` lookup keys to actual image paths.

    If the referenced PNG isn't present locally, stream it from Cauldron via
    `fetch_cauldron_image` and cache it under `image_root`.
    """

    if not Path(table_path).exists():
        return {}

    df = pd.read_parquet(
        table_path, columns=["source_dataset", "config", "source_index", "image_path"]
    )
    df = df.dropna(subset=["source_index"])
    df = df.assign(
        dataset_name=df["config"].fillna(
            df["source_dataset"].str.replace("the_cauldron_", "", regex=False)
        ),
        lookup_key=lambda frame: frame["dataset_name"]
        + ":"
        + frame["source_index"].astype(int).astype(str),
        filename=lambda frame: frame["image_path"].apply(lambda p: Path(p).name),
    )

    lookup: dict[str, Path] = {}
    for row in df.itertuples(index=False):
        dataset_name = row.dataset_name
        source_index = int(row.source_index)
        lookup_key = row.lookup_key
        filename = row.filename or f"{source_index}.png"
        resolved_path = (image_root / dataset_name / filename).resolve()
        if not resolved_path.exists():
            downloaded = fetch_and_cache_image(
                source_config=row.config or dataset_name,
                source_index=source_index,
                image_hash=None,
                image_root=image_root,
                destination=resolved_path,
                prefer_local_cache=False,
            )
            if downloaded is None:
                continue
            resolved_path = downloaded
        lookup[lookup_key] = resolved_path
    return lookup


def fetch_and_cache_image(
    source_config: Optional[str],
    source_index: Optional[int],
    image_hash: Optional[str],
    image_root: Path,
    *,
    destination: Optional[Path] = None,
    prefer_local_cache: bool = True,
) -> Optional[Path]:
    """
    Retrieve the Cauldron image (streaming if necessary) and cache it locally.
    """

    if not source_config or source_index is None:
        return None

    cache_dir = image_root / source_config
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = image_hash if image_hash else f"{int(source_index)}"
    target_path = destination or (cache_dir / f"{filename}.png").resolve()

    if target_path.exists():
        return target_path

    if fetch_cauldron_image is None:
        logger.warning(
            "fetch_cauldron_image unavailable; cannot download %s/%s", source_config, source_index
        )
        return None

    try:
        image, _ = fetch_cauldron_image(
            source_config=source_config,
            source_index=int(source_index),
            image_hash=image_hash,
            prefer_local_cache=prefer_local_cache,
            image_root=image_root,
        )
    except Exception as exc:  # pragma: no cover - network/IO failures
        logger.warning(
            "Failed to fetch Cauldron image %s/%s: %s", source_config, source_index, exc
        )
        return None

    try:
        image.save(target_path)
    except Exception as exc:  # pragma: no cover - PIL save errors
        logger.warning("Failed to save image %s: %s", target_path, exc)
        return None

    return target_path


def resolve_row_image_path(
    row: pd.Series, lookup_map: dict[str, Path], config: ExperimentConfig
) -> Optional[Path]:
    """Resolve the most reliable on-disk image path for a dataset row."""

    candidates: list[Path] = []

    def _append_candidate(value: Any) -> None:
        if isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                path = (config.project_root / path).resolve()
            candidates.append(path)

    _append_candidate(row.get("image_path"))
    _append_candidate(row.get("image_path_absolute"))

    cache_root = row.get("image_cache_root")
    asset = row.get("cauldron_image_asset")
    if isinstance(cache_root, str) and isinstance(asset, str):
        cache_path = Path(cache_root) / asset
        candidates.append(cache_path)
        normalized = Path(str(cache_path).replace("code_base/which_vlm/dataset_builder/dataset", "dataset"))
        if normalized != cache_path:
            candidates.append(normalized)

    lookup_key = row.get("cauldron_lookup_key")
    if isinstance(lookup_key, str) and lookup_key in lookup_map:
        return lookup_map[lookup_key]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: fetch directly from Cauldron if metadata is available.
    source_config = row.get("source_config")
    source_index = row.get("source_index")
    image_hash = row.get("image_bytes_hash")
    downloaded = fetch_and_cache_image(
        source_config=source_config,
        source_index=source_index,
        image_hash=image_hash,
        image_root=config.image_root,
        prefer_local_cache=True,
    )
    if downloaded:
        return downloaded
    return None


@lru_cache(maxsize=2048)
def encode_image_to_data_url(image_path: Path) -> str:
    """Convert an image path into a base64 data URL."""

    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_vlm_messages(
    prompt: str, image_path: Path, system_prompt: Optional[str] = None
) -> list[dict[str, Any]]:
    """Build OpenAI/Qwen-compatible multimodal chat payloads."""

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}},
            ],
        }
    )
    return messages


def prepare_samples(config: ExperimentConfig) -> pd.DataFrame:
    """Load raw router data and return the deduplicated + image-resolved subset."""

    raw_df = pd.read_parquet(config.dataset_path)
    print(raw_df.columns.tolist())
    lookup = build_cauldron_lookup(config.cauldron_lookup_path, config.image_root)
    print(f"Cauldron lookup has {len(lookup)} entries.")
    samples_df = raw_df.drop_duplicates(subset=["sample_id"]).copy()
    print(f"Loaded {len(samples_df)} unique samples from dataset.")
    samples_df["resolved_image_path"] = samples_df.apply(lambda row: resolve_row_image_path(row, lookup, config), axis=1)
    samples_df = samples_df.dropna(subset=["prompt_formatted", "resolved_image_path"]).reset_index(drop=True)

    if config.max_samples is not None:
        n = min(config.max_samples, len(samples_df))
        samples_df = samples_df.sample(n=n, random_state=config.seed).reset_index(drop=True)

    return samples_df


async def evaluate_samples(
    agent: CascadeAgent, df: pd.DataFrame, config: ExperimentConfig
) -> list[dict[str, Any]]:
    """Drive CascadeAgent over the dataframe and capture telemetry."""

    records: list[dict[str, Any]] = []
    iterator = tqdm(df.itertuples(index=False), total=len(df), desc="Routing samples")

    for row in iterator:
        prompt = getattr(row, "prompt_formatted", None) or getattr(row, "prompt_raw", "")
        image_path = Path(row.resolved_image_path)
        system_prompt = getattr(row, "system_prompt", None)
        messages = build_vlm_messages(prompt, image_path, system_prompt)

        row_record: dict[str, Any] = {
            "sample_id": row.sample_id,
            "router_task": getattr(row, "router_task", None),
            "prompt_text": prompt,
            "system_prompt": system_prompt,
            "image_path": str(image_path),
        }

        try:
            cascade_result = await agent.run(
                query=prompt,
                max_tokens=config.generation_max_tokens,
                temperature=config.generation_temperature,
                messages=messages,
            )
            cost_saved = getattr(cascade_result, "cost_saved", None)
            record = {
                **row_record,
                "model_used": cascade_result.model_used,
                "total_cost": cascade_result.total_cost,
                "draft_cost": getattr(cascade_result, "draft_cost", None),
                "verifier_cost": getattr(cascade_result, "verifier_cost", None),
                "cost_saved": cost_saved,
                "savings_pct": _compute_savings_pct(cascade_result.total_cost, cost_saved),
                "cascaded": cascade_result.cascaded,
                "draft_accepted": cascade_result.draft_accepted,
                "routing_strategy": cascade_result.routing_strategy,
                "routing_reason": cascade_result.reason,
                "latency_ms": cascade_result.latency_ms,
                "quality_score": cascade_result.quality_score,
                "quality_threshold": cascade_result.quality_threshold,
                "raw_response": cascade_result.content,
                "metadata": json.dumps(cascade_result.metadata or {}),
                "error": None,
                "exception_type": None,
            }
        except Exception as exc:  # pragma: no cover - transport errors handled at runtime
            record = {
                **row_record,
                "model_used": None,
                "total_cost": None,
                "draft_cost": None,
                "verifier_cost": None,
                "cost_saved": None,
                "savings_pct": None,
                "cascaded": None,
                "draft_accepted": None,
                "routing_strategy": None,
                "routing_reason": None,
                "latency_ms": None,
                "quality_score": None,
                "quality_threshold": None,
                "raw_response": None,
                "metadata": json.dumps({}),
                "error": str(exc),
                "exception_type": exc.__class__.__name__,
            }
        records.append(record)

    return records


def _compute_savings_pct(total_cost: Optional[float], cost_saved: Optional[float]) -> Optional[float]:
    """Helper for cost-derived percentages."""

    if total_cost is None or cost_saved is None:
        return None
    denom = total_cost + cost_saved
    if denom <= 0:
        return None
    return cost_saved / denom


async def run_experiment_async(config: ExperimentConfig) -> pd.DataFrame:
    """Convenience coroutine that prepares data, runs the agent, and returns a DataFrame."""

    samples_df = prepare_samples(config)
    agent = build_cascade_agent(config)
    records = await evaluate_samples(agent, samples_df, config)
    return pd.DataFrame(records)


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """
    Synchronous wrapper around `run_experiment_async`.

    Detects whether an event loop is already running (e.g., inside a notebook) and
    reuses it via `nest_asyncio` so callers can simply invoke this function.
    """

    async def _runner() -> pd.DataFrame:
        return await run_experiment_async(config)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_runner())

    try:
        import nest_asyncio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "An event loop is already running. Install `nest_asyncio` or "
            "call `await run_experiment_async(config)` instead."
        ) from exc

    nest_asyncio.apply()
    return loop.run_until_complete(_runner())


def summarize_results(results_df: pd.DataFrame) -> dict[str, Any]:
    """Compute simple aggregation tables for quick inspection."""

    summary: dict[str, Any] = {}
    if results_df.empty:
        return summary

    successful = results_df[results_df["error"].isna()].copy()
    failed = results_df[results_df["error"].notna()].copy()

    if not successful.empty:
        model_mix = successful["model_used"].value_counts().rename_axis("model").reset_index(name="count")
        model_mix["share_pct"] = (model_mix["count"] / len(successful) * 100).round(2)
        summary["model_mix"] = model_mix

        cost_summary = (
            successful.groupby("model_used")["total_cost"]
            .agg(["count", "sum", "mean"])
            .rename(columns={"sum": "total_cost", "mean": "avg_cost"})
        )
        summary["cost_summary"] = cost_summary

        latency_summary = successful["latency_ms"].describe(percentiles=[0.5, 0.9, 0.95])
        summary["latency_summary"] = latency_summary

    if not failed.empty:
        summary["failures"] = failed[["sample_id", "exception_type", "error"]]

    return summary


def persist_results(results_df: pd.DataFrame, config: ExperimentConfig) -> dict[str, Path]:
    """Write CSV + Parquet outputs for downstream analysis."""

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"{config.experiment_name}_{timestamp}"
    parquet_path = config.output_dir / f"{experiment_id}.parquet"
    csv_path = config.output_dir / f"{experiment_id}.csv"

    results_df.to_parquet(parquet_path, index=False)
    results_df.to_csv(csv_path, index=False)

    return {"parquet": parquet_path, "csv": csv_path}


def run_and_save(config: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """One-shot helper: run experiment, summarize, and persist outputs."""

    results_df = run_experiment(config)
    summary = summarize_results(results_df)
    paths = persist_results(results_df, config)
    return results_df, summary, paths


__all__ = [
    "DEFAULT_MODEL_SPECS",
    "ExperimentConfig",
    "ModelSpec",
    "build_cascade_agent",
    "build_vlm_messages",
    "create_default_config",
    "prepare_samples",
    "evaluate_samples",
    "run_experiment_async",
    "run_experiment",
    "run_and_save",
    "summarize_results",
    "persist_results",
]
