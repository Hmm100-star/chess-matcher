-- Analytics config columns
-- Adds per-classroom analytics win weight and per-assignment-type analytics weight.
-- These are independent of the pairing weights used during round generation.

ALTER TABLE classrooms ADD COLUMN analytics_win_weight INTEGER DEFAULT 50;
ALTER TABLE assignment_types ADD COLUMN analytics_weight INTEGER DEFAULT 50;
