-- Add evaluation columns to vlm_responses table

ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS glider_score FLOAT;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS glider_reasoning TEXT;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS glider_highlight TEXT; -- Storing list as text or JSON

ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS judge_internvl_score FLOAT;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS judge_internvl_rank_group INTEGER;
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS judge_internvl_raw JSONB;

-- Optional text judge
ALTER TABLE vlm_responses ADD COLUMN IF NOT EXISTS judge_text_score FLOAT;
