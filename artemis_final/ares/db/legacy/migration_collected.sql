-- Migration: Create tables for collected data (live traffic)
-- Run this on Postgres startup via docker-entrypoint-initdb.d/

-- ============================================================================
-- vlm_samples_collected: Stores each incoming request
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

CREATE INDEX IF NOT EXISTS idx_samples_collected_request_id ON vlm_samples_collected(request_id);
CREATE INDEX IF NOT EXISTS idx_samples_collected_created_at ON vlm_samples_collected(created_at);

-- ============================================================================
-- vlm_responses_collected: Stores model responses for each sample
-- ============================================================================
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

CREATE INDEX IF NOT EXISTS idx_responses_collected_sample_id ON vlm_responses_collected(sample_id);

-- ============================================================================
-- vlm_feedback: Stores user feedback on samples
-- ============================================================================
CREATE TABLE IF NOT EXISTS vlm_feedback (
    id BIGSERIAL PRIMARY KEY,
    sample_id BIGINT NOT NULL REFERENCES vlm_samples_collected(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    feedback_params JSONB
);

CREATE INDEX IF NOT EXISTS idx_feedback_sample_id ON vlm_feedback(sample_id);
