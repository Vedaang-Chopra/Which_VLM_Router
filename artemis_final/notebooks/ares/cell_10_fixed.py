# === Cell 10: Response Recovery Execution ===
# APPROACH: 
# 1. Fetch sample data from DB 
# 2. If image is missing from DB, fetch from Cauldron
# 3. Run VLM inference with full GPU metrics collection

from ares.db.operations import insert_responses, insert_images
from ares.evaluation.evaluation import Scorer
from ares.configs.db_config import MODEL_PREFIXES, MODEL_NAMES
from ares.data.dataset_loader import CauldronLoader
from ares.evaluation.sample_processor import image_to_png_bytes, compute_image_hash
from PIL import Image
import io

recovery_logger = logging.getLogger('RECOVERY')

def fetch_image_from_cauldron(source_config: str, sample_idx: int, sample_id: str) -> Optional[bytes]:
    """Fetch image from Cauldron if not in DB."""
    try:
        samples = CauldronLoader.load_samples(source_config, n_samples=sample_idx + 10, random_sample=False)
        if sample_idx < len(samples):
            qa = CauldronLoader.extract_qa(samples[sample_idx], source_config)
            if qa and qa.get('image'):
                image = qa['image']
                image_bytes = image_to_png_bytes(image)
                
                # Insert into DB for future use
                image_hash = compute_image_hash(image_bytes)
                image_id = f"img_{image_hash}"
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
                insert_images([image_record])
                recovery_logger.info(f"  📷 Fetched and stored image from Cauldron for {sample_id}")
                return image_bytes
    except Exception as e:
        recovery_logger.error(f"Failed to fetch image from Cauldron for {sample_id}: {e}")
    return None


def process_sample_from_db(
    sample_id: str,
    prompt: str, 
    ground_truth: str,
    image_bytes: bytes,
    source_config: str,
    models_to_run: List[str],
    vlm_client,
    gpu_client,
    scorer,
    config,
    model_specs: List,
) -> List[Dict]:
    """
    Process a sample from DB data with full GPU metrics collection.
    Mirrors the response-building logic from process_sample_normalized.
    """
    from ares.evaluation.confidence import estimate_confidence
    
    response_records = []
    
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Determine which models to process
    models_to_process = [(p, m) for p, m in zip(MODEL_PREFIXES, MODEL_NAMES) if m in models_to_run]
    
    # Run inference for each model
    for model_prefix, model_name in models_to_process:
        try:
            # Get GPU metrics BEFORE inference
            gpu_before = gpu_client.get_metrics(model_name) if gpu_client else {}
            
            # Run inference
            resp = vlm_client.vlm.run_image(
                image=image,
                text=prompt,
                models=[model_name],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            # Get GPU metrics AFTER inference
            gpu_after = gpu_client.get_metrics(model_name) if gpu_client else {}
            
            if model_name not in resp:
                recovery_logger.warning(f"[{model_name}] No response received")
                continue
            
            model_resp = resp[model_name]
            response_text = model_resp.get('response', '')
            metadata = model_resp.get('metadata', {})
            
            # Score the response
            if scorer:
                scores = scorer.score(response_text, ground_truth)
            else:
                scores = {}
            
            # Estimate confidence
            conf_result = estimate_confidence(
                response_text=response_text,
                model_name=model_name,
            )
            
            # Get model spec for cost estimation
            model_spec = next((s for s in model_specs if s.get('name') == model_name), {})
            input_cost = model_spec.get('input_cost_per_1k', 0) / 1000
            output_cost = model_spec.get('output_cost_per_1k', 0) / 1000
            input_tokens = metadata.get('prompt_tokens', 0) or 0
            output_tokens = metadata.get('completion_tokens', 0) or 0
            estimated_cost = (input_tokens * input_cost) + (output_tokens * output_cost)
            
            # Build response record with FULL fields
            record = {
                'sample_id': sample_id,
                'model_name': model_name,
                'model_prefix': model_prefix,
                'model_id': MODEL_NAMES.index(model_name) if model_name in MODEL_NAMES else 0,
                
                # Response content
                'response_raw': response_text,
                'response_parsed': response_text,
                'response_length_chars': len(response_text),
                'response_length_tokens': output_tokens,
                
                # Token counts
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': metadata.get('total_tokens', input_tokens + output_tokens),
                
                # Latency
                'latency_ms': metadata.get('latency_ms', 0),
                
                # Status
                'ok': True,
                'error_message': None,
                'stop_reason': metadata.get('finish_reason', 'stop'),
                'is_refusal': False,
                
                # Confidence
                'confidence_score': conf_result.get('score'),
                'confidence_source': conf_result.get('source'),
                'confidence_reason': conf_result.get('reason'),
                
                # Scoring metrics
                'score_exact_match': scores.get('exact_match', 0.0),
                'score_exact_match_normalized': scores.get('exact_match_normalized', 0.0),
                'score_f1': scores.get('f1', 0.0),
                'score_contains_gt': scores.get('contains_gt', 0.0),
                'score_gt_in_response': scores.get('gt_in_response', 0.0),
                'score_numeric_match': scores.get('numeric_match', 0.0),
                'score_mc_letter_match': scores.get('mc_letter_match', 0.0),
                'is_correct': scores.get('exact_match_normalized', 0.0) > 0.5,
                'pred_answer_letter': None,
                
                # Cost
                'estimated_cost_usd': estimated_cost,
                
                # GPU metrics (from after-inference snapshot)
                'gpu_name': gpu_after.get('gpu_name'),
                'gpu_index': gpu_after.get('gpu_index'),
                'gpu_util_percent': gpu_after.get('gpu_utilization'),
                'gpu_mem_used_mb': gpu_after.get('memory_used_mb'),
                'gpu_mem_total_mb': gpu_after.get('memory_total_mb'),
                'gpu_mem_free_mb': gpu_after.get('memory_free_mb'),
                'gpu_temp_celsius': gpu_after.get('temperature'),
                'gpu_power_watts': gpu_after.get('power_usage'),
                'gpu_power_limit_watts': gpu_after.get('power_limit'),
                'gpu_memory_util_percent': gpu_after.get('memory_utilization'),
                
                # Inference settings
                'inference_temperature': config.temperature,
                'inference_max_tokens': config.max_tokens,
                'inference_top_p': 1.0,
            }
            response_records.append(record)
            
        except Exception as e:
            recovery_logger.error(f"[{model_name}] Inference error: {e}")
            continue
    
    return response_records


# === MAIN RECOVERY LOGIC ===

if config.dry_run:
    print("⏭️ Skipping response recovery (dry run)")
    print(f"   Would attempt to recover up to {min(len(incomplete_df), config.max_recovery_samples)} samples")
elif len(incomplete_df) == 0:
    print("✅ No missing responses to recover!")
elif vlm_client is None:
    print("❌ Cannot recover responses: VLM clients not initialized")
else:
    print(f"🔧 Recovering responses for up to {config.max_recovery_samples} samples...")
    print("=" * 70)
    
    samples_to_recover = incomplete_df.head(config.max_recovery_samples)
    sample_ids = samples_to_recover['sample_id'].tolist()
    
    # Fetch sample data from DB (prompt, ground_truth, image) - LEFT JOIN to catch missing images
    query = """
    SELECT s.sample_id, s.prompt_text, s.ground_truth, s.source_config, s.source_index, i.image_bytes
    FROM vlm_samples s
    LEFT JOIN vlm_images i ON s.image_id = i.image_id
    WHERE s.sample_id = ANY(:sample_ids)
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query), {'sample_ids': sample_ids})
        db_samples = {row[0]: {
            'prompt': row[1], 
            'gt': row[2], 
            'config': row[3], 
            'source_idx': row[4],
            'image': row[5]  # May be None if image missing
        } for row in result}
    
    # Count samples with/without images
    with_image = sum(1 for s in db_samples.values() if s['image'] is not None)
    without_image = len(db_samples) - with_image
    print(f"   📦 Loaded {len(db_samples)} samples from database")
    print(f"   🖼️  {with_image} have images, {without_image} need images from Cauldron")
    
    total_recovered = 0
    total_errors = 0
    response_batch = []
    
    for _, row in tqdm(samples_to_recover.iterrows(), total=len(samples_to_recover), desc="Recovering"):
        sample_id = row['sample_id']
        models_present = row['models_present'] if row['models_present'] else []
        models_missing = [m for m in MODEL_NAMES if m not in models_present]
        
        if not models_missing:
            continue
            
        if sample_id not in db_samples:
            recovery_logger.warning(f"Sample {sample_id} not found in database")
            total_errors += 1
            continue
        
        sample_data = db_samples[sample_id]
        
        # Check if image exists, if not fetch from Cauldron
        image_bytes = sample_data['image']
        if image_bytes is None:
            # Extract index from sample_id (format: {config}_{idx}_{hash})
            try:
                idx = int(sample_id.split('_')[1]) if '_' in sample_id else sample_data.get('source_idx', 0)
            except:
                idx = sample_data.get('source_idx', 0) or 0
            
            image_bytes = fetch_image_from_cauldron(sample_data['config'], idx, sample_id)
            if image_bytes is None:
                recovery_logger.error(f"Could not obtain image for {sample_id}")
                total_errors += 1
                continue
        
        recovery_logger.info(f"[{sample_data['config']}] {sample_id}: recovering {models_missing}")
        
        try:
            records = process_sample_from_db(
                sample_id=sample_id,
                prompt=sample_data['prompt'],
                ground_truth=sample_data['gt'],
                image_bytes=image_bytes,
                source_config=sample_data['config'],
                models_to_run=models_missing,
                vlm_client=vlm_client,
                gpu_client=gpu_client,
                scorer=scorer,
                config=config,
                model_specs=model_specs,
            )
            
            if records:
                response_batch.extend(records)
                total_recovered += len(records)
                recovery_logger.info(f"  ✓ Recovered {len(records)} responses")
            
            # Batch insert
            if len(response_batch) >= config.batch_size:
                insert_responses(response_batch)
                recovery_logger.info(f"  💾 Batch inserted {len(response_batch)} responses")
                response_batch = []
                
        except Exception as e:
            recovery_logger.error(f"Error processing {sample_id}: {e}")
            total_errors += 1
    
    # Final batch
    if response_batch:
        insert_responses(response_batch)
        recovery_logger.info(f"  💾 Final batch inserted {len(response_batch)} responses")
    
    print("\n" + "=" * 70)
    print(f"📊 Response Recovery Complete:")
    print(f"   ✅ Recovered: {total_recovered} responses")
    print(f"   ❌ Errors: {total_errors}")
