-- Run this in your database to add the new columns:
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS confidence_source VARCHAR(50);
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS confidence_reason TEXT;

-- Optional: Drop old columns if they exist
ALTER TABLE vlm_responses DROP COLUMN IF EXISTS confidence_json;
ALTER TABLE vlm_responses DROP COLUMN IF EXISTS gpu_metrics_json;
