
-- Add Molmo evaluation columns to vlm_evaluations
ALTER TABLE vlm_evaluations ADD COLUMN IF NOT EXISTS judge_molmo_score FLOAT;
ALTER TABLE vlm_evaluations ADD COLUMN IF NOT EXISTS judge_molmo_rank_group INTEGER;
ALTER TABLE vlm_evaluations ADD COLUMN IF NOT EXISTS judge_molmo_raw TEXT;
