-- Ares Consolidated Schema (v2.0)
-- 
-- One single file to provision the entire database from scratch.
-- Includes:
-- 1. Core Tables (samples, images, responses, evaluations)
-- 2. Live Traffic Tables (collected samples/responses)
-- 3. Feedback & Error Tracking
-- 4. All columns from all previous migrations

-- ============================================================================
-- 1. CLEANUP (Be careful with production data!)
-- ============================================================================
-- DROP TABLE IF EXISTS retraining_runs CASCADE;
-- DROP TABLE IF EXISTS routing_errors CASCADE;
-- DROP TABLE IF EXISTS vlm_feedback CASCADE;
-- DROP TABLE IF EXISTS vlm_responses_collected CASCADE;
-- DROP TABLE IF EXISTS vlm_samples_collected CASCADE;
-- DROP TABLE IF EXISTS vlm_evaluations CASCADE;
-- DROP TABLE IF EXISTS vlm_responses CASCADE;
-- DROP TABLE IF EXISTS vlm_images CASCADE;
-- DROP TABLE IF EXISTS vlm_samples CASCADE;

-- ============================================================================
-- 2. CORE DATASET: SAMPLES & IMAGES
-- ============================================================================

CREATE TABLE IF NOT EXISTS vlm_samples (
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
    
    -- Link to image
    image_id VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS vlm_images (
    -- Primary Key
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
-- 3. CORE DATASET: RESPONSES & EVALUATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS vlm_responses (
    -- Composite Primary Key identifier
    response_id SERIAL PRIMARY KEY,
    sample_id VARCHAR(255) NOT NULL REFERENCES vlm_samples(sample_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_prefix VARCHAR(10) NOT NULL,
    
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
    
    -- Confidence
    confidence_score FLOAT,
    confidence_source VARCHAR(50),  -- 'logprobs' or 'heuristic'
    confidence_reason TEXT,
    
    -- Rule-based Scores
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
    
    -- GPU Metrics
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
    
    -- Computed scores
    sample_score FLOAT,
    perf_hier FLOAT,
    cost_norm FLOAT,
    utility FLOAT,
    
    -- Evaluation Columns (Migrated in)
    glider_score FLOAT,
    glider_reasoning TEXT,
    glider_highlight TEXT,
    
    judge_internvl_score FLOAT,
    judge_internvl_rank_group INTEGER,
    judge_internvl_raw JSONB,
    judge_text_score FLOAT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vlm_evaluations (
    evaluation_id SERIAL PRIMARY KEY,
    sample_id VARCHAR(255) NOT NULL REFERENCES vlm_samples(sample_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    
    UNIQUE(sample_id, model_name),
    
    response_id INTEGER REFERENCES vlm_responses(response_id),
    
    -- Glider (Legacy/Redundant but kept for compatibility)
    glider_score FLOAT,
    glider_reasoning TEXT,
    glider_highlight TEXT,
    glider_raw_output TEXT,
    
    -- Molmo Judge (Migrated in)
    judge_molmo_score FLOAT,
    judge_molmo_rank_group INTEGER,
    judge_molmo_raw TEXT,
    
    -- Semantic F1
    semantic_f1_precision FLOAT,
    semantic_f1_recall FLOAT,
    semantic_f1_f1 FLOAT,
    semantic_f1_gen_statements JSONB,
    semantic_f1_gt_statements JSONB,
    semantic_f1_matches JSONB,
    semantic_f1_labels JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 4. LIVE TRAFFIC & COLLECTED DATA
-- ============================================================================

CREATE TABLE IF NOT EXISTS vlm_samples_collected (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    router_mode TEXT,
    input_messages JSONB,
    chosen_model TEXT,
    router_decision JSONB,
    lb_decision JSONB,
    meta JSONB
);

CREATE TABLE IF NOT EXISTS vlm_responses_collected (
    id BIGSERIAL PRIMARY KEY,
    sample_id BIGINT NOT NULL REFERENCES vlm_samples_collected(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    model_name TEXT,
    raw_response JSONB,
    normalized_output JSONB,
    latency_ms INTEGER,
    cost_cents NUMERIC(10, 4),
    score FLOAT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS vlm_feedback (
    id BIGSERIAL PRIMARY KEY,
    sample_id BIGINT NOT NULL REFERENCES vlm_samples_collected(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    feedback_params JSONB
);

-- ============================================================================
-- 5. ERROR TRACKING & RETRAINING
-- ============================================================================

CREATE TABLE IF NOT EXISTS routing_errors (
    id SERIAL PRIMARY KEY,
    
    -- Reference
    sample_id INTEGER,  -- FK to vlm_samples_collected
    request_id TEXT,
    
    -- Decision
    router_mode TEXT NOT NULL,
    chosen_model TEXT NOT NULL,
    best_model TEXT,
    
    -- Context
    confidence FLOAT,
    rewards JSONB,
    
    -- Error classification
    error_type TEXT NOT NULL DEFAULT 'unknown',
    severity TEXT DEFAULT 'medium',
    
    -- Impact
    expected_latency_ms FLOAT,
    actual_latency_ms FLOAT,
    expected_cost_usd FLOAT,
    actual_cost_usd FLOAT,
    expected_accuracy FLOAT,
    actual_accuracy FLOAT,
    
    -- Metadata
    task_type TEXT,
    source_dataset TEXT,
    prompt_snippet TEXT,
    meta JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Retraining
    used_for_retraining BOOLEAN DEFAULT FALSE,
    retrained_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS retraining_runs (
    id SERIAL PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    status TEXT DEFAULT 'running',
    
    -- Training details
    num_error_samples INTEGER,
    num_correct_samples INTEGER,
    epochs INTEGER,
    batch_size INTEGER,
    learning_rate FLOAT,
    
    -- Results
    old_checkpoint_path TEXT,
    new_checkpoint_path TEXT,
    metrics_before JSONB,
    metrics_after JSONB,
    improvement JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- 6. INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_samples_source_config ON vlm_samples(source_config);
CREATE INDEX IF NOT EXISTS idx_samples_router_task ON vlm_samples(router_task);
CREATE INDEX IF NOT EXISTS idx_samples_data_split ON vlm_samples(data_split);
CREATE INDEX IF NOT EXISTS idx_samples_run_id ON vlm_samples(run_id);
CREATE INDEX IF NOT EXISTS idx_images_hash ON vlm_images(image_hash);

CREATE INDEX IF NOT EXISTS idx_responses_sample_id ON vlm_responses(sample_id);
CREATE INDEX IF NOT EXISTS idx_responses_model_name ON vlm_responses(model_name);
CREATE INDEX IF NOT EXISTS idx_responses_is_correct ON vlm_responses(is_correct);

CREATE INDEX IF NOT EXISTS idx_evaluations_sample_id ON vlm_evaluations(sample_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_model_name ON vlm_evaluations(model_name);

CREATE INDEX IF NOT EXISTS idx_samples_collected_request_id ON vlm_samples_collected(request_id);
CREATE INDEX IF NOT EXISTS idx_samples_collected_created_at ON vlm_samples_collected(created_at);
CREATE INDEX IF NOT EXISTS idx_responses_collected_sample_id ON vlm_responses_collected(sample_id);
CREATE INDEX IF NOT EXISTS idx_feedback_sample_id ON vlm_feedback(sample_id);

CREATE INDEX IF NOT EXISTS idx_routing_errors_not_retrained ON routing_errors(created_at) WHERE used_for_retraining = FALSE;
CREATE INDEX IF NOT EXISTS idx_routing_errors_type ON routing_errors(error_type, severity);

-- ============================================================================
-- 7. REUSABLE VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW vlm_full_data AS
SELECT 
    s.*,
    i.image_bytes,
    i.image_hash,
    r.model_name,
    r.response_raw,
    r.latency_ms,
    r.is_correct,
    r.confidence_score,
    r.glider_score, -- Now in vlm_responses
    r.judge_internvl_score,
    e.semantic_f1_f1,
    e.judge_molmo_score
FROM vlm_samples s
LEFT JOIN vlm_images i ON s.image_id = i.image_id
LEFT JOIN vlm_responses r ON s.sample_id = r.sample_id
LEFT JOIN vlm_evaluations e ON s.sample_id = e.sample_id AND r.model_name = e.model_name;

CREATE OR REPLACE VIEW routing_errors_summary AS
SELECT 
    DATE_TRUNC('hour', created_at) AS hour,
    router_mode,
    error_type,
    COUNT(*) AS error_count,
    AVG(confidence) AS avg_confidence
FROM routing_errors
GROUP BY DATE_TRUNC('hour', created_at), router_mode, error_type
ORDER BY hour DESC, error_count DESC;

