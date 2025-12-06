-- SQL Schema for Router Live Logs Table
--
-- This table stores all router decisions for offline analysis and retraining.
--
-- Run this SQL to create the table:
--   psql -U vlmrouter -d vlmrouter -f router_logs_schema.sql

-- Drop table if exists (be careful in production!)
-- DROP TABLE IF EXISTS router_live_logs;

CREATE TABLE IF NOT EXISTS router_live_logs (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- Timestamps
    timestamp DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Sample identification
    sample_id VARCHAR(255) NOT NULL,
    source VARCHAR(50) NOT NULL,  -- 'db', 'http', 'synthetic'
    split VARCHAR(50),             -- 'train', 'val', 'test', etc.

    -- Sample content
    text TEXT NOT NULL,
    image_uri TEXT,
    label TEXT,

    -- Router decision
    router_chosen_model VARCHAR(100) NOT NULL,
    router_probs JSONB NOT NULL,
    router_inference_ms DOUBLE PRECISION NOT NULL,

    -- Additional metadata
    extra_metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_router_logs_sample_id ON router_live_logs(sample_id);
CREATE INDEX IF NOT EXISTS idx_router_logs_source ON router_live_logs(source);
CREATE INDEX IF NOT EXISTS idx_router_logs_split ON router_live_logs(split);
CREATE INDEX IF NOT EXISTS idx_router_logs_chosen_model ON router_live_logs(router_chosen_model);
CREATE INDEX IF NOT EXISTS idx_router_logs_timestamp ON router_live_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_router_logs_created_at ON router_live_logs(created_at);

-- GIN index for JSONB columns (for efficient querying)
CREATE INDEX IF NOT EXISTS idx_router_logs_probs ON router_live_logs USING GIN (router_probs);
CREATE INDEX IF NOT EXISTS idx_router_logs_metadata ON router_live_logs USING GIN (extra_metadata);

-- Comments
COMMENT ON TABLE router_live_logs IS 'Logs of all router routing decisions for analysis and retraining';
COMMENT ON COLUMN router_live_logs.timestamp IS 'Unix timestamp when routing decision was made';
COMMENT ON COLUMN router_live_logs.sample_id IS 'Unique identifier for the sample';
COMMENT ON COLUMN router_live_logs.source IS 'Source of sample: db, http, or synthetic';
COMMENT ON COLUMN router_live_logs.split IS 'Dataset split (train/val/test) if from DB';
COMMENT ON COLUMN router_live_logs.text IS 'Question/prompt text';
COMMENT ON COLUMN router_live_logs.image_uri IS 'Path or URL to image';
COMMENT ON COLUMN router_live_logs.label IS 'Ground truth label (if available)';
COMMENT ON COLUMN router_live_logs.router_chosen_model IS 'Model selected by router';
COMMENT ON COLUMN router_live_logs.router_probs IS 'Probability distribution over all models (JSON)';
COMMENT ON COLUMN router_live_logs.router_inference_ms IS 'Router inference latency in milliseconds';
COMMENT ON COLUMN router_live_logs.extra_metadata IS 'Additional metadata as JSON';

-- Example queries:

-- Get router accuracy for DB samples with labels
-- SELECT
--     router_chosen_model,
--     COUNT(*) as count,
--     AVG(CASE WHEN router_chosen_model = label THEN 1 ELSE 0 END) as accuracy
-- FROM router_live_logs
-- WHERE source = 'db' AND label IS NOT NULL
-- GROUP BY router_chosen_model;

-- Get model usage distribution
-- SELECT
--     router_chosen_model,
--     COUNT(*) as count,
--     ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
-- FROM router_live_logs
-- GROUP BY router_chosen_model
-- ORDER BY count DESC;

-- Get latency statistics by source
-- SELECT
--     source,
--     COUNT(*) as samples,
--     ROUND(AVG(router_inference_ms), 2) as avg_latency_ms,
--     ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY router_inference_ms), 2) as p50_ms,
--     ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY router_inference_ms), 2) as p95_ms,
--     ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY router_inference_ms), 2) as p99_ms
-- FROM router_live_logs
-- GROUP BY source;

-- Get samples where router was uncertain (max prob < 0.6)
-- SELECT
--     sample_id,
--     router_chosen_model,
--     router_probs,
--     text
-- FROM router_live_logs
-- WHERE (
--     SELECT MAX(value::numeric)
--     FROM jsonb_each_text(router_probs)
-- ) < 0.6
-- LIMIT 100;
