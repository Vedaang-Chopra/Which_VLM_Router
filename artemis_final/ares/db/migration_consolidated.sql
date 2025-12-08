-- Ares Consolidated Migration
-- 
-- Run this script if you have an existing database and want to bring it up to date
-- with the full schema. It is idempotent (safe to run multiple times).

-- 1. Add new columns to vlm_responses
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS confidence_source VARCHAR(50);
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS confidence_reason TEXT;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS glider_score FLOAT;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS glider_reasoning TEXT;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS glider_highlight TEXT; 
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS judge_internvl_score FLOAT;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS judge_internvl_rank_group INTEGER;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS judge_internvl_raw JSONB;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS judge_text_score FLOAT;

-- 2. Add new columns to vlm_evaluations
ALTER TABLE vlm_evaluations ADD COLUMN IF NOT EXISTS judge_molmo_score FLOAT;
ALTER TABLE vlm_evaluations ADD COLUMN IF NOT EXISTS judge_molmo_rank_group INTEGER;
ALTER TABLE vlm_evaluations ADD COLUMN IF NOT EXISTS judge_molmo_raw TEXT;

-- 3. Create Collected Data Tables
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

-- 4. Create Error Tracking Tables
CREATE TABLE IF NOT EXISTS routing_errors (
    id SERIAL PRIMARY KEY,
    sample_id INTEGER,
    request_id TEXT,
    router_mode TEXT NOT NULL,
    chosen_model TEXT NOT NULL,
    best_model TEXT,
    confidence FLOAT,
    rewards JSONB,
    error_type TEXT NOT NULL DEFAULT 'unknown',
    severity TEXT DEFAULT 'medium',
    expected_latency_ms FLOAT,
    actual_latency_ms FLOAT,
    expected_cost_usd FLOAT,
    actual_cost_usd FLOAT,
    expected_accuracy FLOAT,
    actual_accuracy FLOAT,
    task_type TEXT,
    source_dataset TEXT,
    prompt_snippet TEXT,
    meta JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    used_for_retraining BOOLEAN DEFAULT FALSE,
    retrained_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS retraining_runs (
    id SERIAL PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    status TEXT DEFAULT 'running',
    num_error_samples INTEGER,
    num_correct_samples INTEGER,
    epochs INTEGER,
    batch_size INTEGER,
    learning_rate FLOAT,
    old_checkpoint_path TEXT,
    new_checkpoint_path TEXT,
    metrics_before JSONB,
    metrics_after JSONB,
    improvement JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Create Indexes
CREATE INDEX IF NOT EXISTS idx_samples_collected_request_id ON vlm_samples_collected(request_id);
CREATE INDEX IF NOT EXISTS idx_routing_errors_not_retrained ON routing_errors(created_at) WHERE used_for_retraining = FALSE;

