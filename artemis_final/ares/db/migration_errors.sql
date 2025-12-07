-- Migration: Add routing errors tracking tables
-- Purpose: Track routing misclassifications for targeted retraining

-- Table for tracking routing errors/misroutes
CREATE TABLE IF NOT EXISTS routing_errors (
    id SERIAL PRIMARY KEY,
    
    -- Reference to the original request
    sample_id INTEGER,  -- FK to vlm_samples_collected if from live traffic
    request_id TEXT,    -- UUID of the original request
    
    -- Routing decision details
    router_mode TEXT NOT NULL,            -- accuracy, cheap, fast, balanced
    chosen_model TEXT NOT NULL,           -- Model that was chosen
    best_model TEXT,                      -- Ground truth best model (if known)
    
    -- Confidence and rewards at decision time
    confidence FLOAT,                      -- Router confidence score
    rewards JSONB,                         -- Full rewards dict
    
    -- Error classification
    error_type TEXT NOT NULL DEFAULT 'unknown',  -- misroute, low_confidence, timeout, etc.
    severity TEXT DEFAULT 'medium',              -- low, medium, high, critical
    
    -- Performance impact
    expected_latency_ms FLOAT,
    actual_latency_ms FLOAT,
    expected_cost_usd FLOAT,
    actual_cost_usd FLOAT,
    expected_accuracy FLOAT,
    actual_accuracy FLOAT,
    
    -- Context
    task_type TEXT,
    source_dataset TEXT,
    prompt_snippet TEXT,  -- First 200 chars of prompt for debugging
    
    -- Metadata
    meta JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Retraining status
    used_for_retraining BOOLEAN DEFAULT FALSE,
    retrained_at TIMESTAMP WITH TIME ZONE
);

-- Index for efficient querying during retraining
CREATE INDEX IF NOT EXISTS idx_routing_errors_not_retrained 
ON routing_errors(created_at) 
WHERE used_for_retraining = FALSE;

CREATE INDEX IF NOT EXISTS idx_routing_errors_type 
ON routing_errors(error_type, severity);

CREATE INDEX IF NOT EXISTS idx_routing_errors_model 
ON routing_errors(chosen_model, best_model);


-- Table for tracking retraining runs
CREATE TABLE IF NOT EXISTS retraining_runs (
    id SERIAL PRIMARY KEY,
    
    -- Run metadata
    run_id TEXT UNIQUE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    status TEXT DEFAULT 'running',  -- running, completed, failed
    
    -- Training details
    num_error_samples INTEGER,      -- How many error samples used
    num_correct_samples INTEGER,    -- How many correct samples used
    epochs INTEGER,
    batch_size INTEGER,
    learning_rate FLOAT,
    
    -- Results
    old_checkpoint_path TEXT,
    new_checkpoint_path TEXT,
    
    -- Metrics before/after
    metrics_before JSONB,  -- {accuracy, avg_reward, misroute_rate}
    metrics_after JSONB,
    improvement JSONB,     -- {accuracy_delta, reward_delta, ...}
    
    -- Error info if failed
    error_message TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retraining_runs_status 
ON retraining_runs(status, started_at);


-- View for quick analysis of routing errors
CREATE OR REPLACE VIEW routing_errors_summary AS
SELECT 
    DATE_TRUNC('hour', created_at) AS hour,
    router_mode,
    error_type,
    COUNT(*) AS error_count,
    AVG(confidence) AS avg_confidence,
    COUNT(DISTINCT chosen_model) AS num_models_chosen,
    COUNT(DISTINCT best_model) AS num_best_models
FROM routing_errors
GROUP BY DATE_TRUNC('hour', created_at), router_mode, error_type
ORDER BY hour DESC, error_count DESC;


-- View for model-level error analysis
CREATE OR REPLACE VIEW model_error_rates AS
SELECT 
    chosen_model,
    COUNT(*) AS total_errors,
    COUNT(*) FILTER (WHERE error_type = 'misroute') AS misroutes,
    COUNT(*) FILTER (WHERE error_type = 'low_confidence') AS low_confidence,
    AVG(confidence) AS avg_confidence,
    AVG(actual_latency_ms) AS avg_latency_ms
FROM routing_errors
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY chosen_model
ORDER BY total_errors DESC;
