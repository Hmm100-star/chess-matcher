-- Migration: replace missing_homework_policy string with missing_score_pct float
-- NULL  = exclude (no effect on rating)
-- 0     = treat missing as 0 % correct
-- 1-100 = treat missing as that percentage correct

ALTER TABLE rounds ADD COLUMN missing_score_pct REAL;

-- Backfill existing rows from the old string policy:
--   'exclude' -> NULL (already NULL after ADD COLUMN)
--   'zero' or 'penalty' -> 0.0
UPDATE rounds
SET missing_score_pct = 0.0
WHERE missing_homework_policy IN ('zero', 'penalty');
