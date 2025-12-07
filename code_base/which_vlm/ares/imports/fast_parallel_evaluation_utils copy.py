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

from config import SampleRecord
from dataset_loader import CauldronLoader
from evaluation import Scorer
from modules import FeatureExtractor
from evaluation import GliderEvaluator  # already importing Scorer

REQUEST_TIMEOUT = 60
DEFAULT_EVALUATOR_PORT = 8805
ENABLE_SEMANTIC_F1 = False  # Control flag for semantic F1 evaluation
ENABLE_GLIDER_EVAL = True  # Control flag for Glider evaluation
VLLM_HOST = "http://localhost"  # Change if vLLM is on a different host
METRICS_URL = f"{VLLM_HOST}:10000/metrics"


def return_model_pricing():
    MODEL_PRICING = {
    "deepseek_ocr": {
        "prompt_per_1k": 0.00003,
        "completion_per_1k": 0.0001,
    },
    "qwen2_5_vl_3b": {
        "prompt_per_1k": 0.0001,
        "completion_per_1k": 0.0001,
    },
    "qwen2_5_vl_7b": {
        "prompt_per_1k": 0.0002,
        "completion_per_1k": 0.0002,
    },
    "qwen3_vl_8b_thinking": {
        "prompt_per_1k": 0.00018,
        "completion_per_1k": 0.0021,
    },
    "gemma_3_27b": {
        "prompt_per_1k": 0.00009,
        "completion_per_1k": 0.00016,
    },
    }
    return MODEL_PRICING
MODEL_PRICING = return_model_pricing()

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


def fetch_remote_system_metrics(metrics_url: str, timeout: float = 2.0) -> Dict[str, float]:
    """
    Fetch GPU / system metrics from a remote HTTP endpoint running on the vLLM host.

    The endpoint is expected to return a JSON object with numeric values, e.g.:

        {
            "gpu_index": 0,
            "gpu_mem_total_mb": 24576,
            "gpu_mem_used_mb": 1234,
            "gpu_mem_free_mb": 23342,
            "gpu_utilization_pct": 12,
            "gpu_mem_utilization_pct": 7
        }

    This helper is deliberately forgiving: on any error it just returns {}.
    """
    if not metrics_url:
        return {}

    try:
        resp = requests.get(metrics_url, timeout=timeout)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        # Only keep numeric or simple values so that downstream code is robust
        cleaned: Dict[str, float] = {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                cleaned[k] = float(v)
        return cleaned
    except Exception:
        return {}


def make_vllm_request(
    port: int,
    model_id: str,
    messages: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
    request_timeout: Optional[int] = None,
    logprobs: int = 5,
    metrics_url: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Optional[float]]]]:
    """Make request to a vLLM endpoint and return text + rich metadata.

    In addition to tokens and latency, this computes:
      - token-level logprob aggregates (avg/min/max/std/perplexity,...)
      - simple response-shape features (length, #words, #sentences, error-ish flags)
      - optional remote GPU / system metrics from the vLLM host.
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    payload: Dict = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Ask vLLM for token-level logprobs if requested
    if logprobs and logprobs > 0:
        payload["logprobs"] = logprobs
        payload["top_logprobs"] = logprobs

    # Optional: snapshot system metrics *before* the request
    pre_sys: Dict[str, float] = {}
    if metrics_url:
        pre_sys = fetch_remote_system_metrics(metrics_url)

    try:
        start_time = time.time()
        timeout = request_timeout if request_timeout is not None else REQUEST_TIMEOUT
        response = requests.post(url, json=payload, timeout=timeout)
        latency_ms = (time.time() - start_time) * 1000.0

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None, None

        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]

        usage = data.get("usage", {}) or {}
        finish_reason = choice.get("finish_reason")
        created_ts = data.get("created")
        api_model = data.get("model")  # may or may not be present

        metadata: Dict[str, Optional[float]] = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_ms": latency_ms,
        }

        # Non-numeric metadata that is still useful to propagate
        metadata_str: Dict[str, Optional[str]] = {
            "finish_reason": finish_reason,
            "api_model": api_model,
            "api_created": str(created_ts) if created_ts is not None else None,
        }

        # ---- Confidence & logprob-based aggregates ----
        avg_logprob: Optional[float] = None
        avg_prob: Optional[float] = None
        min_logprob: Optional[float] = None
        max_logprob: Optional[float] = None
        std_logprob: Optional[float] = None
        sum_logprob: Optional[float] = None
        first_token_logprob: Optional[float] = None
        last_token_logprob: Optional[float] = None
        nll_per_token: Optional[float] = None
        perplexity: Optional[float] = None

        token_logprobs: List[float] = []
        logprobs_block = choice.get("logprobs")

        if logprobs_block:
            # vLLM newer format: logprobs["content"] is a list of token entries
            if isinstance(logprobs_block, dict) and "content" in logprobs_block:
                for entry in logprobs_block["content"]:
                    lp = entry.get("logprob")
                    if lp is not None:
                        token_logprobs.append(float(lp))
            # Fallback for older formats
            elif isinstance(logprobs_block, dict) and "token_logprobs" in logprobs_block:
                for lp in logprobs_block.get("token_logprobs", []):
                    if lp is not None:
                        token_logprobs.append(float(lp))

        if token_logprobs:
            import math as _math

            sum_logprob = float(sum(token_logprobs))
            avg_logprob = sum_logprob / len(token_logprobs)
            avg_prob = float(_math.exp(avg_logprob))
            min_logprob = float(min(token_logprobs))
            max_logprob = float(max(token_logprobs))
            first_token_logprob = float(token_logprobs[0])
            last_token_logprob = float(token_logprobs[-1])

            if len(token_logprobs) > 1:
                mean = avg_logprob
                var = sum((lp - mean) ** 2 for lp in token_logprobs) / len(token_logprobs)
                std_logprob = float(_math.sqrt(var))
            else:
                std_logprob = 0.0

            nll_per_token = float(-avg_logprob)
            perplexity = float(_math.exp(-avg_logprob))

        metadata.update(
            {
                "avg_logprob": avg_logprob,
                "avg_prob": avg_prob,
                "min_logprob": min_logprob,
                "max_logprob": max_logprob,
                "std_logprob": std_logprob,
                "sum_logprob": sum_logprob,
                "first_token_logprob": first_token_logprob,
                "last_token_logprob": last_token_logprob,
                "nll_per_token": nll_per_token,
                "perplexity": perplexity,
            }
        )

        # ---- Simple response-shape / quality heuristics ----
        if text is None:
            text_for_stats = ""
        else:
            text_for_stats = text

        num_chars = len(text_for_stats)
        num_words = len(text_for_stats.split())
        num_sentences = sum(text_for_stats.count(sep) for sep in [".", "!", "?"])
        is_empty = len(text_for_stats.strip()) == 0
        looks_like_error = any(
            kw in text_for_stats
            for kw in ["Traceback", "Exception:", "Error:", "ValueError", "KeyError"]
        )
        truncated = finish_reason == "length"

        metadata.update(
            {
                "response_num_chars": float(num_chars),
                "response_num_words": float(num_words),
                "response_num_sentences": float(num_sentences),
                "response_is_empty": float(1.0 if is_empty else 0.0),
                "response_looks_like_error": float(1.0 if looks_like_error else 0.0),
                "response_truncated": float(1.0 if truncated else 0.0),
            }
        )

        # ---- Remote system metrics AFTER the request ----
        if metrics_url:
            post_sys = fetch_remote_system_metrics(metrics_url)
            for k, v in post_sys.items():
                metadata[f"sys_{k}_post"] = v

            # Optional deltas vs pre-request snapshot
            for k, pre_val in pre_sys.items():
                post_val = post_sys.get(k)
                if isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)):
                    metadata[f"sys_{k}_delta"] = float(post_val - pre_val)

        # Merge non-numeric metadata-as-strings at the end
        # (callers can choose to keep/use them or not)
        for k, v in metadata_str.items():
            # Store under a separate namespace to avoid type confusion
            metadata[f"meta_{k}"] = v  # type: ignore[assignment]

        return text, metadata

    except Exception as exc:  # pragma: no cover - network call
        print(f"Request failed: {exc}")
        return None, None


def format_vlm_message(image: Image.Image, prompt: str) -> List[Dict]:
    """Format a multimodal prompt for vLLM-compatible VLMs."""
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

    messages = [
        {"role": "system", "content": "You are an information extraction assistant."},
        {"role": "user", "content": prompt},
    ]

    port = evaluator_port if evaluator_port is not None else DEFAULT_EVALUATOR_PORT
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
                return [str(s).strip() for s in statements if str(s).strip()]
        except Exception:
            pass

    # Simple fallback: split by periods
    return [s.strip() for s in text.split(".") if s.strip()]


def compute_semantic_f1(
    generated: str,
    ground_truth: str,
    evaluator_port: Optional[int] = None,
) -> Dict[str, float]:
    """Compute semantic F1 score similar to Molmo cap F1."""
    gen_statements = extract_atomic_statements(generated, evaluator_port)
    gt_statements = extract_atomic_statements(ground_truth, evaluator_port)

    gen_set = set(gen_statements)
    gt_set = set(gt_statements)

    if not gen_set and not gt_set:
        return {
            "semantic_precision": 1.0,
            "semantic_recall": 1.0,
            "semantic_f1": 1.0,
        }
    if not gen_set:
        return {
            "semantic_precision": 0.0,
            "semantic_recall": 0.0,
            "semantic_f1": 0.0,
        }

    tp = len(gen_set & gt_set)
    fp = len(gen_set - gt_set)
    fn = len(gt_set - gen_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "semantic_precision": precision,
        "semantic_recall": recall,
        "semantic_f1": f1,
    }


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

        # --- Scoring ---
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

        if ENABLE_GLIDER_EVAL and gt_type == "freeform":
            try:
                eval_result = GLIDER_EVAL.evaluate_sample(
                    question=prompt,
                    prediction=response,
                    ground_truth=ground_truth,
                    router_task=router_task,
                )
                glider_score = eval_result.get("score")
                glider_reasoning = eval_result.get("reasoning")
                glider_highlight = eval_result.get("highlight")
                glider_raw_output = eval_result.get("raw_output")
            except Exception as exc:
                print(f"Glider evaluation failed for {sample_id}: {exc}")

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
            response_length_tokens=metadata.get("output_tokens") if metadata else None,
            stop_reason=None,
            error_message=None,
            ok=True,
            **scores,
            **semantic_metrics,
            input_tokens=metadata.get("input_tokens") if metadata else None,
            output_tokens=metadata.get("output_tokens") if metadata else None,
            total_tokens=metadata.get("total_tokens") if metadata else None,
            latency_ms=metadata.get("latency_ms") if metadata else None,
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
                run_id,
            ): model["name"]
            for model in models
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as exc:
                print(f"Model batch failed for {model_name}: {exc}")

    return all_records


def process_config(
    config_name: str,
    models: List[Dict],
    n_samples: int,
    run_id: str,
    output_dir: Path,
    batch_size: int,
    max_workers_batches: int,
) -> pd.DataFrame:
    """
    Process a single Cauldron config: load samples, run all models, save results.

    This function is intended to be run in a separate process for config-level
    parallelism, but it also works fine in-process for debugging or batch processing
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
                    print(f"Batch starting at {start_idx} failed: {exc}")
                finally:
                    pbar.update(1)

    if all_records:
        df = pd.DataFrame([asdict(r) for r in all_records])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{run_id}_{config_name}.parquet"
        df.to_parquet(output_file, index=False)
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
    - Level 2: Batch-level parallelism within each config (ProcessPoolExecutor)
    - Level 3: Model-level parallelism within each batch (ThreadPoolExecutor)
    """
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
                    models,
                    n_samples,
                    run_id,
                    output_dir,
                    batch_size,
                    max_workers_batches,
                ): config_name
                for config_name in configs
            }

            with tqdm(total=len(configs), desc="Configs", leave=True) as pbar:
                for future in as_completed(futures):
                    cfg_name = futures[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            all_dfs.append(df)
                    except Exception as exc:
                        print(f"Config {cfg_name} failed: {exc}")
                    finally:
                        pbar.update(1)
    else:
        # Process configs sequentially (useful for debugging)
        for config_name in tqdm(configs, desc="Configs", leave=True):
            df = process_config(
                config_name=config_name,
                models=models,
                n_samples=n_samples,
                run_id=run_id,
                output_dir=output_dir,
                batch_size=batch_size,
                max_workers_batches=max_workers_batches,
            )
            if not df.empty:
                all_dfs.append(df)

    if all_dfs:
        results_df = pd.concat(all_dfs, ignore_index=True)
        combined_file = output_dir / f"{run_id}_all_results.parquet"
        results_df.to_parquet(combined_file, index=False)
        print(f"\nSaved combined results to {combined_file}")
        return results_df

    print("No results were generated.")
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
