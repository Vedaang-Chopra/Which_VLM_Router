import base64
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

from artemis_final.ares.configs.config import SampleRecord
from artemis_final.ares.data.dataset_loader import CauldronLoader
from artemis_final.ares.evaluation.evaluation import Scorer
from modules import FeatureExtractor

REQUEST_TIMEOUT = 60
DEFAULT_EVALUATOR_PORT = 8805
ENABLE_SEMANTIC_F1 = False  # Control flag for semantic F1 evaluation
ENABLE_GLIDER_EVAL = True  # Control flag for Glider evaluation
from artemis_final.ares.evaluation.evaluation import GliderEvaluator  # already importing Scorer

def _glider_chat_fn(messages, max_tokens, model_name):
    # We ignore model_name except for routing; here it should be "PatronusAI/glider"
    text, _ = make_vllm_request(
        port=DEFAULT_EVALUATOR_PORT,
        model_id=model_name,
        messages=messages,
        max_tokens=max_tokens,
        request_timeout=300,
    )
    return text or ""
    
# Create a single global evaluator instance to avoid re-instantiation overhead
GLIDER_EVAL = GliderEvaluator(
    chat_fn=_glider_chat_fn,
    model_name="PatronusAI/glider",
)


def configure(
    *,
    request_timeout: Optional[int] = None,
    evaluator_port: Optional[int] = None,
    enable_semantic_f1: Optional[bool] = None,
    enable_glider_eval: Optional[bool] = None,
) -> None:
    """Update module-level defaults for request timeout, evaluator port, and evaluation flags."""
    global REQUEST_TIMEOUT, DEFAULT_EVALUATOR_PORT, ENABLE_SEMANTIC_F1, ENABLE_GLIDER_EVAL
    if request_timeout is not None:
        REQUEST_TIMEOUT = request_timeout
    if evaluator_port is not None:
        DEFAULT_EVALUATOR_PORT = evaluator_port
    if enable_semantic_f1 is not None:
        ENABLE_SEMANTIC_F1 = enable_semantic_f1
    if enable_glider_eval is not None:
        ENABLE_GLIDER_EVAL = enable_glider_eval


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def compute_image_hash(image: Image.Image) -> str:
    """Compute SHA256 hash of image."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return hashlib.sha256(buffered.getvalue()).hexdigest()[:16]


def make_vllm_request(
    port: int,
    model_id: str,
    messages: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
    request_timeout: Optional[int] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Optional[float]]]]:
    """Make request to a vLLM endpoint."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        start_time = time.time()
        timeout = request_timeout if request_timeout is not None else REQUEST_TIMEOUT
        response = requests.post(url, json=payload, timeout=timeout)
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            metadata = {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "latency_ms": latency_ms,
            }
            return text, metadata

        print(f"Error {response.status_code}: {response.text}")
        return None, None
    except Exception as exc:  # pragma: no cover - network call
        print(f"Request failed: {exc}")
        return None, None


def format_vlm_message(image: Image.Image, prompt: str) -> List[Dict]:
    """Format message for VLM with image."""
    image_b64 = image_to_base64(image)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def extract_atomic_statements(text: str, evaluator_port: Optional[int] = None) -> List[str]:
    """Use Glider to extract atomic statements from text."""
    prompt = (
        "Extract all distinct atomic statements from the following text.\n"
        "Return ONLY a JSON array of strings, one per statement.\n\n"
        f"Text: {text}\n\nJSON array:"
    )

    port = evaluator_port if evaluator_port is not None else DEFAULT_EVALUATOR_PORT
    messages = [{"role": "user", "content": prompt}]
    response, _ = make_vllm_request(
        port=port,
        model_id="PatronusAI/glider",
        messages=messages,
        max_tokens=1024,
    )

    if response:
        try:
            statements = json.loads(response)
            if isinstance(statements, list):
                return statements
        except Exception:
            pass

    return [s.strip() for s in text.split(".") if s.strip()]


def compute_semantic_f1(
    generated: str,
    ground_truth: str,
    evaluator_port: Optional[int] = None,
) -> Dict[str, float]:
    """Compute semantic F1 score similar to Molmo cap F1."""
    gen_statements = extract_atomic_statements(generated, evaluator_port)
    gt_statements = extract_atomic_statements(ground_truth, evaluator_port)

    if not gen_statements or not gt_statements:
        return {"semantic_precision": 0.0, "semantic_recall": 0.0, "semantic_f1": 0.0}

    matches = 0
    for gen_stmt in gen_statements:
        prompt = (
            "Is the following generated statement consistent with the ground truth?\n"
            "Answer with ONLY 'yes' or 'no'.\n\n"
            f"Generated statement: {gen_stmt}\n"
            f"Ground truth: {ground_truth}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        port = evaluator_port if evaluator_port is not None else DEFAULT_EVALUATOR_PORT
        response, _ = make_vllm_request(
            port=port,
            model_id="PatronusAI/glider",
            messages=messages,
            max_tokens=10,
        )

        if response and "yes" in response.lower():
            matches += 1

    precision = matches / len(gen_statements) if gen_statements else 0.0
    recall = matches / len(gt_statements) if gt_statements else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"semantic_precision": precision, "semantic_recall": recall, "semantic_f1": f1}


def process_sample(
    sample: Dict,
    sample_idx: int,
    config_name: str,
    model: Dict,
    run_id: str,
) -> Optional[SampleRecord]:
    """Process a single sample with a single model."""
    try:
        qa_data = CauldronLoader.extract_qa(sample, config_name)
        if not qa_data:
            return None

        image = qa_data["image"]
        prompt = qa_data["prompt"]
        ground_truth = qa_data["ground_truth"]
        router_task = qa_data["router_task"]
        gt_type = qa_data["ground_truth_type"]

        # --- Features & IDs ---
        img_features = FeatureExtractor.extract_image_features(image)
        txt_features = FeatureExtractor.extract_text_features(prompt)
        image_hash = compute_image_hash(image)
        sample_id = f"{config_name}_{sample_idx:05d}_{image_hash}"

        # --- VLM inference ---
        messages = format_vlm_message(image, prompt)
        response, metadata = make_vllm_request(
            port=model["port"],
            model_id=model["id"],
            messages=messages,
        )

        if response is None:
            return None

        metadata = metadata or {}

        # --- Base scalar scores (Scorer) ---
        scores = Scorer.compute_all_scores(
            pred=response,
            gt=ground_truth,
            gt_type=gt_type,
        )

        # --- Semantic F1 (Glider-as-judge, Molmo-style) ---
        semantic_metrics = {
            "semantic_f1_precision": None,
            "semantic_f1_recall": None,
            "semantic_f1_f1": None,
        }
        if ENABLE_SEMANTIC_F1 and gt_type == "freeform":
            try:
                sem = compute_semantic_f1(
                    generated=response,
                    ground_truth=ground_truth,
                    evaluator_port=None,  # uses DEFAULT_EVALUATOR_PORT
                )
                semantic_metrics = {
                    "semantic_f1_precision": sem["semantic_precision"],
                    "semantic_f1_recall": sem["semantic_recall"],
                    "semantic_f1_f1": sem["semantic_f1"],
                }
            except Exception as exc:
                print(f"Semantic F1 evaluation failed for {sample_id}: {exc}")

        # --- Glider rubric score (GliderEvaluator) ---
        glider_score = None
        glider_reasoning = None
        glider_highlight = None
        glider_raw_output = None

        if ENABLE_GLIDER_EVAL and gt_type in ("freeform", "exact", "numeric", "mc"):
            try:
                glider_result = GLIDER_EVAL.evaluate(
                    question=prompt,
                    model_answer=response,
                    ground_truth=ground_truth,
                    sample_id=sample_id,
                )
                glider_score = glider_result["score"]
                glider_reasoning = glider_result["reasoning"]
                glider_highlight = glider_result["highlight"]
                glider_raw_output = glider_result["raw_output"]
            except Exception as exc:
                print(f"Glider evaluation failed for {sample_id}: {exc}")

        # --- Final record ---
        record = SampleRecord(
            sample_id=sample_id,
            run_id=run_id,
            timestamp_utc=datetime.utcnow().isoformat(),
            image_path=None,
            image_bytes_hash=image_hash,
            prompt_raw=prompt,
            prompt_formatted=None,
            system_prompt=None,
            source_dataset=f"cauldron_{config_name}",
            source_config=config_name,
            router_task=router_task,
            ground_truth=ground_truth,
            ground_truth_type=gt_type,
            mc_options=qa_data.get("mc_options"),
            source_index=sample_idx,
            **img_features,
            **txt_features,
            model_name=model["name"],
            model_id=model["id"],
            response_raw=response,
            response_parsed=response,
            response_length_chars=len(response),
            response_length_tokens=metadata.get("output_tokens"),
            stop_reason=None,
            error_message=None,
            ok=True,
            **scores,
            **semantic_metrics,
            input_tokens=metadata.get("input_tokens"),
            output_tokens=metadata.get("output_tokens"),
            total_tokens=metadata.get("total_tokens"),
            latency_ms=metadata.get("latency_ms"),
            estimated_cost_usd=0.0,
            inference_temperature=0.0,
            inference_max_tokens=512,
            inference_top_p=1.0,
            glider_score=glider_score,
            glider_reasoning=glider_reasoning,
            glider_highlight=glider_highlight,
            glider_raw_output=glider_raw_output,
        )
        return record

    except Exception as exc:
        print(f"Error processing sample {sample_idx}: {exc}")
        return None


def process_model_batch(
    samples: List[Dict],
    start_idx: int,
    config_name: str,
    model: Dict,
    run_id: str,
) -> List[SampleRecord]:
    """Process a batch of samples for one model."""
    records: List[SampleRecord] = []
    for local_idx, sample in enumerate(samples):
        global_idx = start_idx + local_idx
        record = process_sample(sample, global_idx, config_name, model, run_id)
        if record:
            records.append(record)
    return records



def process_data_batch(
    samples: List[Dict],
    start_idx: int,
    config_name: str,
    models: List[Dict],
    run_id: str,
) -> List[SampleRecord]:
    """
    Process a single batch of data across all models.
    This function runs in a separate worker process.
    """
    all_records: List[SampleRecord] = []

    # Process all models for this batch in parallel using threads
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                process_model_batch,
                samples,
                start_idx,
                config_name,
                model,
                run_id
            ): model["name"]
            for model in models
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as exc:
                print(f"Model {model_name} failed on batch starting at {start_idx}: {exc}")

    return all_records


def process_config(
    config_name: str,
    n_samples: int,
    models: List[Dict],
    run_id: str,
    output_dir: Path,
    batch_size: int = 8,
    max_workers_batches: int = 4,
) -> pd.DataFrame:
    """
    Process one Cauldron config across all models with batch-level parallelism.

    Args:
        config_name: Name of the Cauldron config
        n_samples: Number of samples to process
        models: List of model configurations
        run_id: Unique run identifier
        output_dir: Directory to save results
        batch_size: Number of samples per batch
        max_workers_batches: Max parallel workers for batch processing
    """
    print(f"\nProcessing config: {config_name}")

    try:
        samples = CauldronLoader.load_samples(config_name, n_samples)
    except Exception as exc:
        print(f"Failed to load {config_name}: {exc}")
        return pd.DataFrame()

    all_records: List[SampleRecord] = []

    # Split samples into batches
    num_batches = (len(samples) + batch_size - 1) // batch_size
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(samples))
        batches.append((samples[start_idx:end_idx], start_idx))

    print(f"  Split into {num_batches} batches of ~{batch_size} samples each")

    # Process batches in parallel across multiple workers
    with ProcessPoolExecutor(max_workers=max_workers_batches) as executor:
        futures = {
            executor.submit(
                process_data_batch,
                batch_samples,
                start_idx,
                config_name,
                models,
                run_id,
            ): (start_idx, len(batch_samples))
            for batch_samples, start_idx in batches
        }

        # Progress bar for batches
        with tqdm(total=num_batches, desc=f"{config_name} (batches)", leave=True) as pbar:
            for future in as_completed(futures):
                start_idx, batch_len = futures[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                except Exception as exc:
                    print(f"Batch starting at {start_idx} failed on {config_name}: {exc}")
                finally:
                    pbar.update(1)

    if all_records:
        df = pd.DataFrame([asdict(r) for r in all_records])
        output_file = output_dir / f"{config_name}.parquet"
        df.to_parquet(output_file)
        print(f"Saved {len(df)} records to {output_file}")
        return df

    return pd.DataFrame()


def run_parallel_evaluation(
    configs: List[str],
    models: List[Dict],
    n_samples: int,
    max_workers: int,
    run_id: str,
    output_dir: Path,
    batch_size: int = 8,
    max_workers_batches: int = 4,
    parallel_configs: bool = True,
) -> pd.DataFrame:
    """
    Run evaluation across all configs with multi-level parallelism.

    Architecture:
    - Level 1: Config-level parallelism (ProcessPoolExecutor, optional)
    - Level 2: Batch-level parallelism (ProcessPoolExecutor within each config)
    - Level 3: Model-level parallelism (ThreadPoolExecutor within each batch)

    Args:
        configs: List of Cauldron config names
        models: List of model configurations
        n_samples: Number of samples per config
        max_workers: Max parallel workers for config processing (Level 1)
        run_id: Unique run identifier
        output_dir: Output directory for results
        batch_size: Number of samples per batch (Level 2)
        max_workers_batches: Max parallel workers for batch processing (Level 2)
        parallel_configs: If True, process configs in parallel; else sequential

    Returns:
        Combined DataFrame with all results
    """
    print(f"\n{'=' * 80}")
    print("Starting parallel evaluation")
    print(f"Configs: {len(configs)}")
    print(f"Models: {len(models)}")
    print(f"Samples per config: {n_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Max parallel configs: {max_workers if parallel_configs else 1}")
    print(f"Max parallel batches per config: {max_workers_batches}")
    print(f"Max parallel models per batch: {len(models)}")
    print(f"Total samples: {len(configs) * n_samples * len(models)}")
    print(f"{'=' * 80}\n")

    all_dfs: List[pd.DataFrame] = []

    if parallel_configs:
        # Process configs in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_config,
                    config_name,
                    n_samples,
                    models,
                    run_id,
                    output_dir,
                    batch_size,
                    max_workers_batches,
                ): config_name
                for config_name in configs
            }

            with tqdm(total=len(configs), desc="Processing configs") as pbar:
                for future in as_completed(futures):
                    config_name = futures[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            all_dfs.append(df)
                    except Exception as exc:
                        print(f"Config {config_name} failed: {exc}")
                    finally:
                        pbar.update(1)
    else:
        # Process configs sequentially (better for debugging)
        for config_name in tqdm(configs, desc="Processing configs"):
            df = process_config(
                config_name=config_name,
                n_samples=n_samples,
                models=models,
                run_id=run_id,
                output_dir=output_dir,
                batch_size=batch_size,
                max_workers_batches=max_workers_batches,
            )
            if not df.empty:
                all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_file = output_dir / "all_results.parquet"
        combined_df.to_parquet(combined_file)
        print(f"\nSaved combined results: {combined_file}")
        print(f"Total records: {len(combined_df)}")
        return combined_df

    return pd.DataFrame()


__all__ = [
    "configure",
    "image_to_base64",
    "compute_image_hash",
    "make_vllm_request",
    "format_vlm_message",
    "extract_atomic_statements",
    "compute_semantic_f1",
    "process_sample",
    "process_model_batch",
    "process_data_batch",
    "process_config",
    "run_parallel_evaluation",
]
