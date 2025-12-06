"""
Sample processor for normalized VLM database schema.

Inserts into separate tables:
- vlm_samples (raw sample data)
- vlm_images (image bytes)
- vlm_responses (model responses per sample)
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import hashlib
from PIL import Image
import io

from ares.evaluation.confidence import estimate_confidence


def extract_mc_letter(text: str) -> Optional[str]:
    """Extract multiple choice letter (A, B, C, D) from response."""
    if not text:
        return None
    patterns = [
        r'^([A-D])\.',
        r'^([A-D])$',
        r'\(([A-D])\)',
        r'[Aa]nswer[:\s]*([A-D])',
        r'^([A-D])\s',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip()[:50])
        if match:
            return match.group(1).upper()
    return None


def parse_response(response_text: str) -> str:
    """Parse/clean response text."""
    if not response_text:
        return ""
    text = response_text.strip()
    for prefix in ["Answer:", "The answer is", "Response:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text


def check_refusal(response_text: str) -> bool:
    """Check if response is a refusal."""
    if not response_text:
        return False
    refusal_phrases = [
        "i cannot", "i can't", "i'm not able", "i am not able",
        "i'm unable", "i am unable", "as an ai", "i don't have",
    ]
    lower = response_text.lower()
    return any(phrase in lower for phrase in refusal_phrases)


def assign_split(sample_id: str, train_ratio: float = 0.70, val_ratio: float = 0.15) -> str:
    """Deterministically assign a sample to train/val/test split."""
    hash_bytes = hashlib.md5(sample_id.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    hash_float = hash_int / (2**64)
    
    if hash_float < train_ratio:
        return 'train'
    elif hash_float < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'


def image_to_png_bytes(pil_image: Image.Image) -> bytes:
    """Convert PIL Image to PNG bytes for database storage."""
    buffer = io.BytesIO()
    if pil_image.mode not in ('RGB', 'L'):
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA256 hash of image bytes, return first 16 chars."""
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def process_sample_normalized(
    sample: Dict,
    sample_idx: int,
    source_config: str,
    vlm_client,
    gpu_client,
    scorer,
    config,
    model_specs: List,
    models_to_run: List[str] = None,  # NEW: Optional list of models to run (skip others)
) -> Optional[Tuple[Dict, Dict, List[Dict]]]:
    """
    Process a single sample and return records for normalized tables.
    
    Args:
        models_to_run: If provided, only run inference for these models.
                      If None, run all models.
    
    Returns:
        Tuple of (sample_record, image_record, list_of_response_records)
        Or None if processing fails.
    """
    try:
        from ares.data.dataset_loader import CauldronLoader
        from ares.configs.db_config import MODEL_PREFIXES, MODEL_NAMES
        
        qa = CauldronLoader.extract_qa(sample, source_config)
        if qa is None:
            return None
        
        image = qa['image']
        prompt = qa['prompt']
        ground_truth = qa['ground_truth']
        router_task = qa['router_task']
        gt_type = qa['ground_truth_type']
        mc_options = qa.get('mc_options')
        
        # Process image
        image_bytes = image_to_png_bytes(image)
        image_hash = compute_image_hash(image_bytes)
        
        # Generate IDs
        sample_id = f"{source_config}_{sample_idx}_{image_hash[:8]}"
        image_id = f"img_{image_hash}"
        
        # Assign split
        data_split = assign_split(sample_id, config.train_ratio, config.val_ratio)
        
        # Extract GT letter for MC
        gt_answer_letter = extract_mc_letter(ground_truth)
        
        # Text features
        txt_prompt_length_chars = len(prompt)
        txt_prompt_length_words = len(prompt.split())
        txt_has_mc_options = bool(mc_options)
        
        # Question type
        txt_question_type = None
        lower_prompt = prompt.lower()
        for qtype in ['what', 'how', 'why', 'which', 'where', 'when', 'who']:
            if qtype in lower_prompt:
                txt_question_type = qtype
                break
        if txt_question_type is None and '?' in prompt:
            txt_question_type = 'other_question'
        
        # ===================
        # TABLE 1: vlm_samples
        # ===================
        sample_record = {
            'sample_id': sample_id,
            'run_id': config.run_id,
            'source_config': source_config,
            'source_dataset': 'cauldron',
            'source_index': sample_idx,
            'router_task': router_task,
            'ground_truth_type': gt_type,
            'data_split': data_split,
            'prompt_text': prompt,
            'prompt_formatted': prompt,
            'system_prompt': None,
            'mc_options': json.dumps(mc_options) if mc_options else None,
            'ground_truth': ground_truth,
            'gt_answer_letter': gt_answer_letter,
            'txt_prompt_length_chars': txt_prompt_length_chars,
            'txt_prompt_length_words': txt_prompt_length_words,
            'txt_question_type': txt_question_type,
            'txt_has_mc_options': txt_has_mc_options,
            'image_id': image_id,
        }
        
        # ===================
        # TABLE 2: vlm_images
        # ===================
        image_record = {
            'image_id': image_id,
            'image_bytes': image_bytes,
            'image_hash': image_hash,
            'img_width': image.width,
            'img_height': image.height,
            'img_aspect_ratio': image.width / image.height if image.height > 0 else 1.0,
            'img_file_size_bytes': len(image_bytes),
            'image_path': None,
            'image_cache_root': None,
            'cauldron_image_asset': None,
            'cauldron_lookup_key': f"{source_config}_{sample_idx}",
        }
        
        # ===================
        # TABLE 3: vlm_responses (one per model)
        # ===================
        response_records = []
        
        # Determine which models to actually process
        if models_to_run is None:
            models_to_process = list(zip(MODEL_PREFIXES, MODEL_NAMES))
        else:
            models_to_process = [(p, m) for p, m in zip(MODEL_PREFIXES, MODEL_NAMES) if m in models_to_run]
        
        # Run inference only for models we need
        if models_to_run is None:
            responses = vlm_client.vlm.run_image(
                image=image,
                text=prompt,
                models="all",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        else:
            responses = {}
            for model_name in models_to_run:
                try:
                    resp = vlm_client.vlm.run_image(
                        image=image,
                        text=prompt,
                        models=[model_name],
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                    responses.update(resp)
                except Exception as e:
                    responses[model_name] = {'ok': False, 'error_message': str(e)}
        
        for prefix, model_name in models_to_process:
            response = responses.get(model_name, {})
            
            # Get GPU metrics
            gpu_summary = gpu_client.get_gpu_summary(model_name)
            
            # Extract response data
            response_text = response.get('response_text', '') or ''
            ok = response.get('ok', False)
            error_msg = response.get('error_message')
            
            # Tokens
            usage = response.get('usage', {})
            input_tokens = response.get('input_tokens') or usage.get('prompt_tokens', 0) or 0
            output_tokens = response.get('output_tokens') or usage.get('completion_tokens', 0) or 0
            total_tokens = input_tokens + output_tokens
            
            latency_ms = response.get('latency_ms', 0) or 0
            stop_reason = response.get('stop_reason')
            est_cost = response.get('est_cost', 0.0) or 0.0
            
            # Parse response
            response_parsed = parse_response(response_text)
            is_refusal = check_refusal(response_text)
            pred_answer_letter = extract_mc_letter(response_text)
            
            # Compute scores
            scores = scorer.compute_all_scores(response_text, ground_truth, gt_type)
            
            # MC letter match
            score_mc_letter_match = None
            if gt_answer_letter and pred_answer_letter:
                score_mc_letter_match = 1.0 if gt_answer_letter == pred_answer_letter else 0.0
            
            # Confidence from logprobs
            conf_score, conf_json = estimate_confidence(response, prefer_logprobs=True)
            
            # Get model spec
            model_spec = {}
            for spec in model_specs:
                if spec.get('name') == model_name:
                    model_spec = spec
                    break
            
            response_record = {
                'sample_id': sample_id,
                'model_name': model_name,
                'model_prefix': prefix,
                'model_id': model_spec.get('model_id', model_name),
                'response_raw': response_text,
                'response_parsed': response_parsed,
                'response_length_chars': len(response_text),
                'response_length_tokens': output_tokens,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'latency_ms': latency_ms,
                'ok': ok,
                'error_message': error_msg,
                'stop_reason': stop_reason,
                'is_refusal': is_refusal,
                'confidence_score': conf_score,
                'confidence_source': conf_json.get('source') if conf_json else None,
                'confidence_reason': conf_json.get('reason') if conf_json else None,
                'score_exact_match': scores['score_exact_match'],
                'score_exact_match_normalized': scores['score_exact_match_normalized'],
                'score_f1': scores['score_f1'],
                'score_contains_gt': scores.get('score_contains_gt', 0.0),
                'score_gt_in_response': scores.get('score_gt_in_response', 0.0),
                'score_numeric_match': scores.get('score_numeric_match'),
                'score_mc_letter_match': score_mc_letter_match,
                'is_correct': scores['is_correct'],
                'pred_answer_letter': pred_answer_letter,
                'estimated_cost_usd': est_cost,
                # GPU metrics (flat columns only)
                'gpu_name': gpu_summary.get('gpu_name') if gpu_summary else None,
                'gpu_index': gpu_summary.get('gpu_index') if gpu_summary else None,
                'gpu_util_percent': gpu_summary.get('util_percent') if gpu_summary else None,
                'gpu_mem_used_mb': gpu_summary.get('mem_used_mb') if gpu_summary else None,
                'gpu_mem_total_mb': gpu_summary.get('mem_total_mb') if gpu_summary else None,
                'gpu_mem_free_mb': gpu_summary.get('mem_free_mb') if gpu_summary else None,
                'gpu_temp_celsius': gpu_summary.get('temp_celsius') if gpu_summary else None,
                'gpu_power_watts': gpu_summary.get('power_watts') if gpu_summary else None,
                'gpu_power_limit_watts': gpu_summary.get('power_limit_watts') if gpu_summary else None,
                'gpu_memory_util_percent': gpu_summary.get('memory_util_percent') if gpu_summary else None,
                # Inference config
                'inference_temperature': config.temperature,
                'inference_max_tokens': config.max_tokens,
                'inference_top_p': getattr(config, 'top_p', 1.0),
            }
            
            response_records.append(response_record)
        
        return sample_record, image_record, response_records
        
    except Exception as e:
        import traceback
        print(f"Error processing sample {sample_idx}: {e}")
        traceback.print_exc()
        return None
