"""Spawn-safe worker that handles a single `source_config` for parallel inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure the repo / package root is on sys.path when this module is imported in a new process.
ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_engine.client import WhichVLMClient
from ares.evaluation.sample_processor import process_sample_normalized
from ares.utils.common_utils import return_model_specs
from ares.data.dataset_loader import CauldronLoader
from ares.evaluation.evaluation import Scorer
from ares.metrics.metrics_client import GPUMetricsClient
from ares.db.operations import insert_samples, insert_images, insert_responses


class _Config:
    """Minimal data holder used inside the worker process."""

    def __init__(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            setattr(self, key, value)


def process_single_config(args: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process all samples for a single Cauldron config via multiprocessing.

    Parameters
    ----------
    args :
        Tuple of (source_config, serialized config dict).
    """
    source_config, config_dict = args
    config = _Config(config_dict)
    status_interval = getattr(config, "status_interval", 10)

    vlm_client = WhichVLMClient.from_yaml(config.models_yaml)
    gpu_client = GPUMetricsClient(endpoints=config.gpu_endpoints)
    model_specs = return_model_specs()
    scorer = Scorer()

    results = {"config": source_config, "processed": 0, "errors": 0}

    try:
        samples = CauldronLoader.load_samples(
            source_config,
            n_samples=config.samples_per_config,
            random_sample=True,
        )
    except Exception as exc:  # pragma: no cover - handles data loading failures
        results["error"] = str(exc)
        return results

    print(f"[{source_config}] Starting processing ({len(samples)} candidates, batch insert={config.batch_insert_size})")

    sample_batch: list[Dict[str, Any]] = []
    image_batch: list[Dict[str, Any]] = []
    response_batch: list[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        try:
            result = process_sample_normalized(
                sample=sample,
                sample_idx=idx,
                source_config=source_config,
                vlm_client=vlm_client,
                gpu_client=gpu_client,
                scorer=scorer,
                config=config,
                model_specs=model_specs,
            )

            if result:
                sample_record, image_record, response_records = result
                sample_batch.append(sample_record)
                image_batch.append(image_record)
                response_batch.extend(response_records)
                results["processed"] += 1

                if len(sample_batch) >= config.batch_insert_size:
                    insert_images(image_batch)
                    insert_samples(sample_batch)
                    insert_responses(response_batch)
                    print(f"[{source_config}] Persisted {results['processed']} samples (batch insert)")
                    sample_batch = []
                    image_batch = []
                    response_batch = []

                if results["processed"] and results["processed"] % status_interval == 0:
                    print(f"[{source_config}] {results['processed']} samples processed so far")
        except Exception:
            results["errors"] += 1
            print(f"[{source_config}] Error encountered (total errors={results['errors']})")

    if sample_batch:
        insert_images(image_batch)
        insert_samples(sample_batch)
        insert_responses(response_batch)

    print(f"[{source_config}] Finished – processed={results['processed']}, errors={results['errors']}")
    return results
