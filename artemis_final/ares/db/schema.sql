-- VLM Router Normalized Schema
-- 4 Tables: samples, images, responses, evaluations

-- ============================================================================
-- TABLE 1: vlm_samples - Raw sample data (prompts, ground truth, metadata)
-- ============================================================================
DROP TABLE IF EXISTS vlm_evaluations CASCADE;
DROP TABLE IF EXISTS vlm_responses CASCADE;
DROP TABLE IF EXISTS vlm_images CASCADE;
DROP TABLE IF EXISTS vlm_samples CASCADE;

CREATE TABLE vlm_samples (
    -- Primary Key
    sample_id VARCHAR(255) PRIMARY KEY,
    
    -- Run metadata
    run_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Source data
    source_config VARCHAR(100) NOT NULL,
    source_dataset VARCHAR(100) DEFAULT 'cauldron',
    source_index INTEGER NOT NULL,
    router_task VARCHAR(100),
    ground_truth_type VARCHAR(50),  -- 'mc', 'numeric', 'freeform', 'exact'
    data_split VARCHAR(20),  -- 'train', 'val', 'test'
    
    -- Prompt data
    prompt_text TEXT NOT NULL,
    prompt_formatted TEXT,
    system_prompt TEXT,
    mc_options JSONB,  -- Array of MC options if applicable
    
    -- Ground truth
    ground_truth TEXT NOT NULL,
    gt_answer_letter VARCHAR(10),  -- Extracted from GT for MC
    
    -- Text features
    txt_prompt_length_chars INTEGER,
    txt_prompt_length_words INTEGER,
    txt_question_type VARCHAR(100),
    txt_has_mc_options BOOLEAN DEFAULT FALSE,
    
    -- Link to image (foreign key)
    image_id VARCHAR(255)
);

-- ============================================================================
-- TABLE 2: vlm_images - Image data (bytes, hash, dimensions)
-- ============================================================================
CREATE TABLE vlm_images (
    -- Primary Key (same as sample_id for 1:1 relationship)
    image_id VARCHAR(255) PRIMARY KEY,
    
    -- Image binary data
    image_bytes BYTEA,
    image_hash VARCHAR(64) NOT NULL,
    
    -- Dimensions
    img_width INTEGER,
    img_height INTEGER,
    img_aspect_ratio FLOAT,
    img_file_size_bytes INTEGER,
    
    -- Optional paths
    image_path TEXT,
    image_cache_root TEXT,
    cauldron_image_asset TEXT,
    cauldron_lookup_key TEXT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- TABLE 3: vlm_responses - Model responses, scores, confidence, GPU metrics
-- One row per (sample_id, model_name) combination
-- ============================================================================
CREATE TABLE vlm_responses (
    -- Composite Primary Key
    response_id SERIAL PRIMARY KEY,
    sample_id VARCHAR(255) NOT NULL REFERENCES vlm_samples(sample_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_prefix VARCHAR(10) NOT NULL,  -- 'm1', 'm2', etc.
    
    -- Unique constraint
    UNIQUE(sample_id, model_name),
    
    -- Model info
    model_id VARCHAR(255),
    
    -- Response data
    response_raw TEXT,
    response_parsed TEXT,
    response_length_chars INTEGER,
    response_length_tokens INTEGER,
    
    -- Tokens and timing
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms FLOAT,
    
    -- Status
    ok BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    stop_reason TEXT,
    is_refusal BOOLEAN DEFAULT FALSE,
    
    -- Confidence (from logprobs)
    confidence_score FLOAT,
    confidence_source VARCHAR(50),  -- 'logprobs' or 'heuristic'
    confidence_reason TEXT,
    -- Note: confidence details removed to avoid duplication
    
    -- Scores
    score_exact_match FLOAT,
    score_exact_match_normalized FLOAT,
    score_f1 FLOAT,
    score_contains_gt FLOAT,
    score_gt_in_response FLOAT,
    score_numeric_match FLOAT,
    score_mc_letter_match FLOAT,
    is_correct BOOLEAN,
    
    -- Predicted answer
    pred_answer_letter VARCHAR(10),
    
    -- Cost
    estimated_cost_usd FLOAT,
    
    -- GPU Metrics (flat columns only, no JSON duplicate)
    gpu_name VARCHAR(100),
    gpu_index INTEGER,
    gpu_util_percent FLOAT,
    gpu_mem_used_mb FLOAT,
    gpu_mem_total_mb FLOAT,
    gpu_mem_free_mb FLOAT,
    gpu_temp_celsius FLOAT,
    gpu_power_watts FLOAT,
    gpu_power_limit_watts FLOAT,
    gpu_memory_util_percent FLOAT,
    
    -- Inference config
    inference_temperature FLOAT,
    inference_max_tokens INTEGER,
    inference_top_p FLOAT,
    
    -- Computed scores (from Notebook 02)
    sample_score FLOAT,
    perf_hier FLOAT,
    cost_norm FLOAT,
    utility FLOAT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- TABLE 4: vlm_evaluations - Glider scores, Semantic F1 (computed later)
-- One row per (sample_id, model_name) combination
-- ============================================================================
CREATE TABLE vlm_evaluations (
    -- Composite Primary Key
    evaluation_id SERIAL PRIMARY KEY,
    sample_id VARCHAR(255) NOT NULL REFERENCES vlm_samples(sample_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    
    -- Unique constraint
    UNIQUE(sample_id, model_name),
    
    -- Reference to response
    response_id INTEGER REFERENCES vlm_responses(response_id),
    
    -- Glider Evaluation
    glider_score FLOAT,
    glider_reasoning TEXT,
    glider_highlight TEXT,
    glider_raw_output TEXT,
    
    -- Semantic F1
    semantic_f1_precision FLOAT,
    semantic_f1_recall FLOAT,
    semantic_f1_f1 FLOAT,
    semantic_f1_gen_statements JSONB,
    semantic_f1_gt_statements JSONB,
    semantic_f1_matches JSONB,
    semantic_f1_labels JSONB,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================
CREATE INDEX idx_samples_source_config ON vlm_samples(source_config);
CREATE INDEX idx_samples_router_task ON vlm_samples(router_task);
CREATE INDEX idx_samples_data_split ON vlm_samples(data_split);
CREATE INDEX idx_samples_run_id ON vlm_samples(run_id);

CREATE INDEX idx_images_hash ON vlm_images(image_hash);

CREATE INDEX idx_responses_sample_id ON vlm_responses(sample_id);
CREATE INDEX idx_responses_model_name ON vlm_responses(model_name);
CREATE INDEX idx_responses_is_correct ON vlm_responses(is_correct);

CREATE INDEX idx_evaluations_sample_id ON vlm_evaluations(sample_id);
CREATE INDEX idx_evaluations_model_name ON vlm_evaluations(model_name);
CREATE INDEX idx_evaluations_glider_score ON vlm_evaluations(glider_score);

-- ============================================================================
-- HELPER VIEW: Join all tables for easy querying
-- ============================================================================
CREATE OR REPLACE VIEW vlm_full_data AS
SELECT 
    s.*,
    i.image_bytes,
    i.image_hash,
    i.img_width,
    i.img_height,
    i.img_aspect_ratio,
    i.img_file_size_bytes,
    r.model_name,
    r.model_prefix,
    r.model_id,
    r.response_raw,
    r.response_parsed,
    r.input_tokens,
    r.output_tokens,
    r.total_tokens,
    r.latency_ms,
    r.ok,
    r.error_message,
    r.confidence_score,
    r.confidence_source,
    r.score_exact_match,
    r.score_f1,
    r.score_mc_letter_match,
    r.is_correct,
    r.pred_answer_letter,
    r.estimated_cost_usd,
    r.gpu_util_percent,
    r.gpu_mem_used_mb,
    r.gpu_temp_celsius,
    r.gpu_power_watts,
    e.glider_score,
    e.glider_reasoning,
    e.semantic_f1_f1
FROM vlm_samples s
LEFT JOIN vlm_images i ON s.image_id = i.image_id
LEFT JOIN vlm_responses r ON s.sample_id = r.sample_id
LEFT JOIN vlm_evaluations e ON s.sample_id = e.sample_id AND r.model_name = e.model_name;
