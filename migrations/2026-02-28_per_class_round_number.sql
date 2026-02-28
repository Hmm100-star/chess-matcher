-- Migration: add per-class round_number to rounds table
-- Each classroom gets its own sequential round counter (1, 2, 3 …).
-- Existing rounds are left as NULL so they gracefully fall back to
-- displaying their global DB id in the UI.
--
-- For PostgreSQL (production):
ALTER TABLE rounds
    ADD COLUMN IF NOT EXISTS round_number INTEGER;

-- For SQLite (local dev), IF NOT EXISTS is not supported.
-- The app.py startup migration path handles SQLite safely via PRAGMA table_info.
