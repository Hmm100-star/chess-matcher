-- Migration: add per-assignment-type missing_score_pct to round_assignment_types
-- Applied: 2026-02-28
ALTER TABLE round_assignment_types
  ADD COLUMN IF NOT EXISTS missing_score_pct FLOAT DEFAULT NULL;
