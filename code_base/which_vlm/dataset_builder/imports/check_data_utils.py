"""
Utilities that help inspect evaluation records in notebooks such as
`02_check_data.ipynb`.

The helpers focus on two workflows:

1. Build a dataframe that gathers every model's output for the same
   underlying image/sample (identified via `sample_id` / `image_bytes_hash`).
2. Retrieve the corresponding Cauldron sample/image so it can be displayed
   while inspecting failures.

Typical usage inside the notebook:

```python
from code_base.which_vlm.dataset_builder import check_data_utils as cdu

df = cdu.load_run_records('./code_base/which_vlm/dataset_builder/experiment_data/runs/exp_XXXX')
image_df = cdu.build_same_image_dataframe(df)
focus = image_df.iloc[0]
img, _ = cdu.fetch_cauldron_image(
    focus.source_config,
    focus.source_index,
    image_hash=focus.image_bytes_hash,
)
display(img)
```
"""

from __future__ import annotations

import hashlib
import itertools
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd

if TYPE_CHECKING:
    from PIL import Image


CAULDRON_REPO = "HuggingFaceM4/the_cauldron"

DEFAULT_IMAGE_ROOT = Path("dataset/which_vlm_data/images/cauldron")


@dataclass
class ModelOutcome:
    """Compact structure that stores per-model evaluation info for a sample."""

    model_name: str
    model_id: str
    is_correct: bool
    score_f1: float
    latency_ms: float
    response_raw: Optional[str]
    glider_score: Optional[float] = None
    glider_reasoning: Optional[str] = None
    glider_highlight: Optional[str] = None

    @classmethod
    def from_row(cls, row: pd.Series) -> "ModelOutcome":
        return cls(
            model_name=row.get("model_name"),
            model_id=row.get("model_id"),
            is_correct=bool(row.get("is_correct", False)),
            score_f1=float(row.get("score_f1", 0.0)),
            latency_ms=float(row.get("latency_ms", 0.0)),
            response_raw=row.get("response_raw"),
            glider_score=row.get("glider_score"),
            glider_reasoning=row.get("glider_reasoning"),
            glider_highlight=row.get("glider_highlight"),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_run_records(
    run_dir: Path | str,
    *,
    subset: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load all per-config parquet files under `run_dir` into a single dataframe.

    Args:
        run_dir: Path to a directory like
            `code_base/which_vlm/dataset_builder/experiment_data/runs/exp_*`.
        subset: Optional iterable of config names to keep (e.g., ["docvqa"]).

    Returns:
        pd.DataFrame with one row per (sample, model) pair.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    parquet_files = sorted(run_dir.glob("*.parquet"))
    if subset:
        subset = set(subset)
        parquet_files = [f for f in parquet_files if f.stem in subset]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {run_dir}")

    frames: List[pd.DataFrame] = []
    for file_path in parquet_files:
        frames.append(pd.read_parquet(file_path))

    return pd.concat(frames, ignore_index=True)


def build_same_image_dataframe(
    df: pd.DataFrame,
    *,
    min_models: int = 2,
) -> pd.DataFrame:
    """
    Collapse the long-form evaluation dataframe into one row per unique sample.

    Args:
        df: Dataframe returned by `load_run_records` (or equivalent).
        min_models: Keep only samples that have outputs from at least
            this many distinct models.

    Returns:
        pd.DataFrame with:
            - sample_id / image_bytes_hash / source_config / source_index
            - prompt / ground truth / router_task
            - n_models, n_correct
            - models (list[str])
            - model_outcomes (list[ModelOutcome] serialized as dicts)
            - model_records (full per-model dictionaries with every column)
    """
    required_cols = {"sample_id", "image_bytes_hash", "source_config", "router_task"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    grouped_records: List[Dict[str, object]] = []
    grouped = df.groupby(
        ["sample_id", "image_bytes_hash", "source_config", "source_index"], sort=False
    )

    for (sample_id, image_hash, source_config, source_index), group in grouped:
        unique_models = group["model_name"].nunique(dropna=True)
        if unique_models < min_models:
            continue

        group = group.sort_values("model_name")
        first_row = group.iloc[0]

        model_outcomes = [ModelOutcome.from_row(row).to_dict() for _, row in group.iterrows()]
        model_records = [row.to_dict() for _, row in group.iterrows()]

        grouped_records.append(
            {
                "sample_id": sample_id,
                "image_bytes_hash": image_hash,
                "source_config": source_config,
                "source_index": int(source_index) if pd.notna(source_index) else None,
                "router_task": first_row.get("router_task"),
                "prompt_raw": first_row.get("prompt_raw"),
                "ground_truth": first_row.get("ground_truth"),
                "n_models": unique_models,
                "n_correct": int(group["is_correct"].sum()),
                "models": group["model_name"].tolist(),
                "model_outcomes": model_outcomes,
                "model_records": model_records,
            }
        )

    return pd.DataFrame(grouped_records)


def find_local_image_path(
    image_hash: str,
    source_config: str,
    *,
    image_root: Path = DEFAULT_IMAGE_ROOT,
) -> Optional[Path]:
    """
    Return the path to the cached PNG saved during dataset building, if present.
    """
    candidate = image_root / source_config / f"{image_hash}.png"
    return candidate if candidate.exists() else None


def load_cauldron_sample(
    source_config: str,
    source_index: int,
) -> Dict:
    """
    Stream the Cauldron dataset until we reach `source_index` for the config.

    The returned dict is identical to what `datasets` yields inside
    `CauldronLoader`.
    """
    from datasets import load_dataset

    if source_index is None:
        raise ValueError("source_index is required to locate the Cauldron sample")

    target_idx = int(source_index)

    ds = load_dataset(
        CAULDRON_REPO,
        source_config,
        streaming=True,
    )
    iterator = ds["train"]
    sample = next(itertools.islice(iterator, target_idx, None), None)
    if sample is None:
        raise IndexError(
            f"Index {source_index} out of range for config '{source_config}'"
        )
    return sample


def fetch_cauldron_image(
    source_config: str,
    source_index: int,
    *,
    image_hash: Optional[str] = None,
    prefer_local_cache: bool = True,
    image_root: Path = DEFAULT_IMAGE_ROOT,
) -> Tuple["Image.Image", Dict]:
    """
    Return a PIL Image for the requested sample plus the raw HF sample dict.

    Order of operations:
        1. If `prefer_local_cache` and `{image_hash}.png` exists locally,
           load it directly.
        2. Otherwise stream-load the Cauldron sample and return its image.

    Raises:
        RuntimeError if the streamed sample does not match `image_hash`
        (when provided).
    """
    from PIL import Image

    if prefer_local_cache and image_hash:
        candidate = find_local_image_path(image_hash, source_config, image_root=image_root)
        if candidate:
            img = Image.open(candidate)
            img.load()
            return img, {"images": [img]}

    sample = load_cauldron_sample(source_config, source_index)
    image: Image.Image = sample["images"][0]

    if image_hash:
        computed_hash = _compute_image_hash(image)
        if computed_hash != image_hash:
            raise RuntimeError(
                f"Hash mismatch for {source_config}/{source_index}: "
                f"{computed_hash} (downloaded) != {image_hash} (recorded)"
            )

    return image, sample


def _compute_image_hash(image: "Image.Image") -> str:
    """Compute the truncated SHA256 hash used in our parquet files."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


__all__ = [
    "ModelOutcome",
    "build_same_image_dataframe",
    "fetch_cauldron_image",
    "find_local_image_path",
    "load_cauldron_sample",
    "load_run_records",
]
