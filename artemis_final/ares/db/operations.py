"""Database CRUD operations for normalized VLM schema."""

import json
from typing import List, Dict, Any, Optional, Tuple, Set
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ares.db.connection import get_engine


def get_existing_responses(
    sample_ids: List[str],
    engine: Engine = None,
) -> Dict[str, Set[str]]:
    """
    Check which (sample_id, model_name) pairs already exist in the database.
    
    Returns:
        Dict mapping sample_id -> set of model_names that have responses
    """
    if not sample_ids:
        return {}
    if engine is None:
        engine = get_engine()
    
    # Query existing responses for these sample_ids
    placeholders = ', '.join([f':id_{i}' for i in range(len(sample_ids))])
    query = text(f"""
        SELECT sample_id, model_name 
        FROM vlm_responses 
        WHERE sample_id IN ({placeholders}) AND ok = true
    """)
    
    params = {f'id_{i}': sid for i, sid in enumerate(sample_ids)}
    
    existing = {}
    with engine.connect() as conn:
        result = conn.execute(query, params)
        for row in result:
            sample_id, model_name = row[0], row[1]
            if sample_id not in existing:
                existing[sample_id] = set()
            existing[sample_id].add(model_name)
    
    return existing


def get_existing_sample_ids(
    sample_ids: List[str],
    engine: Engine = None,
) -> Set[str]:
    """
    Check which sample_ids already exist in vlm_samples.
    
    Returns:
        Set of sample_ids that already exist
    """
    if not sample_ids:
        return set()
    if engine is None:
        engine = get_engine()
    
    placeholders = ', '.join([f':id_{i}' for i in range(len(sample_ids))])
    query = text(f"SELECT sample_id FROM vlm_samples WHERE sample_id IN ({placeholders})")
    params = {f'id_{i}': sid for i, sid in enumerate(sample_ids)}
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        return {row[0] for row in result}


def get_existing_evaluations(
    sample_ids: List[str],
    engine: Engine = None,
) -> Dict[str, Set[str]]:
    """
    Check which (sample_id, model_name) pairs already have evaluations.
    
    Returns:
        Dict mapping sample_id -> set of model_names with evaluations
    """
    if not sample_ids:
        return {}
    if engine is None:
        engine = get_engine()
    
    placeholders = ', '.join([f':id_{i}' for i in range(len(sample_ids))])
    query = text(f"""
        SELECT sample_id, model_name 
        FROM vlm_evaluations 
        WHERE sample_id IN ({placeholders}) AND glider_score IS NOT NULL
    """)
    
    params = {f'id_{i}': sid for i, sid in enumerate(sample_ids)}
    
    existing = {}
    with engine.connect() as conn:
        result = conn.execute(query, params)
        for row in result:
            sample_id, model_name = row[0], row[1]
            if sample_id not in existing:
                existing[sample_id] = set()
            existing[sample_id].add(model_name)
    
    return existing


def get_responses_needing_utility(engine: Engine = None) -> List[str]:
    """
    Get response_ids that need utility computation (utility is NULL).
    
    Returns:
        List of response_ids needing utility update
    """
    if engine is None:
        engine = get_engine()
    
    query = text("""
        SELECT response_id FROM vlm_responses 
        WHERE utility IS NULL AND ok = true
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result]


def insert_samples(
    records: List[Dict[str, Any]],
    engine: Engine = None,
) -> int:
    """Insert sample records into vlm_samples table."""
    if not records:
        return 0
    if engine is None:
        engine = get_engine()
    
    insert_sql = text("""
        INSERT INTO vlm_samples (
            sample_id, run_id, source_config, source_dataset, source_index,
            router_task, ground_truth_type, data_split, prompt_text, prompt_formatted,
            system_prompt, mc_options, ground_truth, gt_answer_letter,
            txt_prompt_length_chars, txt_prompt_length_words, txt_question_type,
            txt_has_mc_options, image_id
        ) VALUES (
            :sample_id, :run_id, :source_config, :source_dataset, :source_index,
            :router_task, :ground_truth_type, :data_split, :prompt_text, :prompt_formatted,
            :system_prompt, :mc_options, :ground_truth, :gt_answer_letter,
            :txt_prompt_length_chars, :txt_prompt_length_words, :txt_question_type,
            :txt_has_mc_options, :image_id
        )
        ON CONFLICT (sample_id) DO UPDATE SET
            run_id = EXCLUDED.run_id,
            updated_at = NOW()
    """)
    
    with engine.begin() as conn:
        for record in records:
            conn.execute(insert_sql, record)
    return len(records)


def insert_images(
    records: List[Dict[str, Any]],
    engine: Engine = None,
) -> int:
    """Insert image records into vlm_images table."""
    if not records:
        return 0
    if engine is None:
        engine = get_engine()
    
    insert_sql = text("""
        INSERT INTO vlm_images (
            image_id, image_bytes, image_hash, img_width, img_height,
            img_aspect_ratio, img_file_size_bytes, image_path, image_cache_root,
            cauldron_image_asset, cauldron_lookup_key
        ) VALUES (
            :image_id, :image_bytes, :image_hash, :img_width, :img_height,
            :img_aspect_ratio, :img_file_size_bytes, :image_path, :image_cache_root,
            :cauldron_image_asset, :cauldron_lookup_key
        )
        ON CONFLICT (image_id) DO NOTHING
    """)
    
    with engine.begin() as conn:
        for record in records:
            conn.execute(insert_sql, record)
    return len(records)


def insert_responses(
    records: List[Dict[str, Any]],
    engine: Engine = None,
) -> int:
    """Insert response records into vlm_responses table."""
    if not records:
        return 0
    if engine is None:
        engine = get_engine()
    
    insert_sql = text("""
        INSERT INTO vlm_responses (
            sample_id, model_name, model_prefix, model_id,
            response_raw, response_parsed, response_length_chars, response_length_tokens,
            input_tokens, output_tokens, total_tokens, latency_ms,
            ok, error_message, stop_reason, is_refusal,
            confidence_score, confidence_source, confidence_reason,
            score_exact_match, score_exact_match_normalized, score_f1,
            score_contains_gt, score_gt_in_response, score_numeric_match,
            score_mc_letter_match, is_correct, pred_answer_letter,
            estimated_cost_usd,
            gpu_name, gpu_index, gpu_util_percent, gpu_mem_used_mb,
            gpu_mem_total_mb, gpu_mem_free_mb, gpu_temp_celsius,
            gpu_power_watts, gpu_power_limit_watts, gpu_memory_util_percent,
            inference_temperature, inference_max_tokens, inference_top_p
        ) VALUES (
            :sample_id, :model_name, :model_prefix, :model_id,
            :response_raw, :response_parsed, :response_length_chars, :response_length_tokens,
            :input_tokens, :output_tokens, :total_tokens, :latency_ms,
            :ok, :error_message, :stop_reason, :is_refusal,
            :confidence_score, :confidence_source, :confidence_reason,
            :score_exact_match, :score_exact_match_normalized, :score_f1,
            :score_contains_gt, :score_gt_in_response, :score_numeric_match,
            :score_mc_letter_match, :is_correct, :pred_answer_letter,
            :estimated_cost_usd,
            :gpu_name, :gpu_index, :gpu_util_percent, :gpu_mem_used_mb,
            :gpu_mem_total_mb, :gpu_mem_free_mb, :gpu_temp_celsius,
            :gpu_power_watts, :gpu_power_limit_watts, :gpu_memory_util_percent,
            :inference_temperature, :inference_max_tokens, :inference_top_p
        )
        ON CONFLICT (sample_id, model_name) DO UPDATE SET
            response_raw = EXCLUDED.response_raw,
            response_parsed = EXCLUDED.response_parsed,
            input_tokens = EXCLUDED.input_tokens,
            output_tokens = EXCLUDED.output_tokens,
            latency_ms = EXCLUDED.latency_ms,
            ok = EXCLUDED.ok,
            confidence_score = EXCLUDED.confidence_score,
            score_exact_match = EXCLUDED.score_exact_match,
            score_f1 = EXCLUDED.score_f1,
            is_correct = EXCLUDED.is_correct,
            estimated_cost_usd = EXCLUDED.estimated_cost_usd,
            gpu_util_percent = EXCLUDED.gpu_util_percent,
            gpu_mem_used_mb = EXCLUDED.gpu_mem_used_mb,
            updated_at = NOW()
    """)

    
    with engine.begin() as conn:
        for record in records:
            conn.execute(insert_sql, record)
    return len(records)


def insert_evaluations(
    records: List[Dict[str, Any]],
    engine: Engine = None,
) -> int:
    """Insert evaluation records into vlm_evaluations table."""
    if not records:
        return 0
    if engine is None:
        engine = get_engine()
    
    insert_sql = text("""
        INSERT INTO vlm_evaluations (
            sample_id, model_name,
            glider_score, glider_reasoning, glider_highlight, glider_raw_output,
            semantic_f1_precision, semantic_f1_recall, semantic_f1_f1,
            semantic_f1_gen_statements, semantic_f1_gt_statements,
            semantic_f1_matches, semantic_f1_labels,
            judge_molmo_score, judge_molmo_rank_group, judge_molmo_raw
        ) VALUES (
            :sample_id, :model_name,
            :glider_score, :glider_reasoning, :glider_highlight, :glider_raw_output,
            :semantic_f1_precision, :semantic_f1_recall, :semantic_f1_f1,
            :semantic_f1_gen_statements, :semantic_f1_gt_statements,
            :semantic_f1_matches, :semantic_f1_labels,
            :judge_molmo_score, :judge_molmo_rank_group, :judge_molmo_raw
        )
        ON CONFLICT (sample_id, model_name) DO UPDATE SET
            glider_score = COALESCE(EXCLUDED.glider_score, vlm_evaluations.glider_score),
            glider_reasoning = COALESCE(EXCLUDED.glider_reasoning, vlm_evaluations.glider_reasoning),
            glider_highlight = COALESCE(EXCLUDED.glider_highlight, vlm_evaluations.glider_highlight),
            judge_molmo_score = COALESCE(EXCLUDED.judge_molmo_score, vlm_evaluations.judge_molmo_score),
            judge_molmo_rank_group = COALESCE(EXCLUDED.judge_molmo_rank_group, vlm_evaluations.judge_molmo_rank_group),
            judge_molmo_raw = COALESCE(EXCLUDED.judge_molmo_raw, vlm_evaluations.judge_molmo_raw),
            updated_at = NOW()
    """)
    
    required_keys = [
        'sample_id', 'model_name',
        'glider_score', 'glider_reasoning', 'glider_highlight', 'glider_raw_output',
        'semantic_f1_precision', 'semantic_f1_recall', 'semantic_f1_f1',
        'semantic_f1_gen_statements', 'semantic_f1_gt_statements',
        'semantic_f1_matches', 'semantic_f1_labels',
        'judge_molmo_score', 'judge_molmo_rank_group', 'judge_molmo_raw'
    ]
    
    with engine.begin() as conn:
        for record in records:
            filled_record = {k: record.get(k) for k in required_keys}
            conn.execute(insert_sql, filled_record)
    return len(records)


# Legacy function for backward compatibility
def batch_insert_records(
    table_name: str,
    records: List[Dict[str, Any]],
    engine: Engine = None,
    batch_size: int = 100
) -> int:
    """Legacy batch insert - use table-specific functions instead."""
    if table_name == 'vlm_samples':
        return insert_samples(records, engine)
    elif table_name == 'vlm_images':
        return insert_images(records, engine)
    elif table_name == 'vlm_responses':
        return insert_responses(records, engine)
    elif table_name == 'vlm_evaluations':
        return insert_evaluations(records, engine)
    else:
        raise ValueError(f"Unknown table: {table_name}")


def get_row_count(table_name: str, where_clause: str = None, engine: Engine = None) -> int:
    """Get row count with optional filter."""
    if engine is None:
        engine = get_engine()
    
    sql = f"SELECT COUNT(*) FROM {table_name}"
    if where_clause:
        sql += f" WHERE {where_clause}"
    
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return result.scalar()


def batch_update_records(
    table_name: str,
    records: List[Dict[str, Any]],
    key_column: str = 'sample_id',
    engine: Engine = None
) -> int:
    """
    Batch update records in any table by key column.
    Each record dict should include the key_column and fields to update.
    """
    if engine is None:
        engine = get_engine()
    
    if not records:
        return 0
    
    count = 0
    for record in records:
        key_value = record.pop(key_column, None)
        if not key_value or not record:
            continue
        
        # Build SET clause
        set_parts = [f"{k} = :{k}" for k in record.keys()]
        set_clause = ", ".join(set_parts)
        
        sql = text(f"""
            UPDATE {table_name}
            SET {set_clause}, updated_at = NOW()
            WHERE {key_column} = :_key_value
        """)
        
        params = {**record, '_key_value': key_value}
        for k, v in params.items():
            if isinstance(v, (dict, list)):
                params[k] = json.dumps(v)
        
        with engine.begin() as conn:
            conn.execute(sql, params)
        count += 1
    
    return count