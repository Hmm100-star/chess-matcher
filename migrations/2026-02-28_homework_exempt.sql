-- 2026-02-28: Add per-student homework exemption flags to homework_entries
ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS white_exempt BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS black_exempt BOOLEAN NOT NULL DEFAULT FALSE;
