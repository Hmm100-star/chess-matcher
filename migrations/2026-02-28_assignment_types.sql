-- Assignment Types system migration
-- Creates four new tables and seeds them from existing homework data.
-- Safe to run on a fresh DB (all statements are idempotent via IF NOT EXISTS / ON CONFLICT).
--
-- Table overview:
--   assignment_types          – classroom-level pool of assignment categories
--   round_assignment_types    – which types are active per round, with weight/total_questions
--   assignment_entries        – per-match per-type score records (replaces homework_entries)
--   student_assignment_scores – cumulative correct/incorrect per student per type
--
-- NOTE: The CREATE TABLE statements are also applied automatically at app startup via
--       POSTGRES_COMPATIBILITY_PATCH_STATEMENTS in db.py.  Run this file manually ONCE on
--       an existing production database to backfill data from legacy homework tables.

-- ── Schema ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS assignment_types (
    id               SERIAL PRIMARY KEY,
    classroom_id     INTEGER NOT NULL REFERENCES classrooms(id) ON DELETE CASCADE,
    name             VARCHAR(200) NOT NULL,
    metric_mode      VARCHAR(20)  NOT NULL DEFAULT 'pct_correct',
    missing_policy   VARCHAR(20)  NOT NULL DEFAULT 'zero',
    created_at       TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS round_assignment_types (
    id                 SERIAL PRIMARY KEY,
    round_id           INTEGER NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
    assignment_type_id INTEGER NOT NULL REFERENCES assignment_types(id) ON DELETE CASCADE,
    weight             INTEGER NOT NULL DEFAULT 30,
    total_questions    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS assignment_entries (
    id                 SERIAL PRIMARY KEY,
    match_id           INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    assignment_type_id INTEGER NOT NULL REFERENCES assignment_types(id) ON DELETE CASCADE,
    white_correct      INTEGER NOT NULL DEFAULT 0,
    white_incorrect    INTEGER NOT NULL DEFAULT 0,
    black_correct      INTEGER NOT NULL DEFAULT 0,
    black_incorrect    INTEGER NOT NULL DEFAULT 0,
    white_submitted    BOOLEAN NOT NULL DEFAULT FALSE,
    black_submitted    BOOLEAN NOT NULL DEFAULT FALSE,
    white_pct_wrong    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    black_pct_wrong    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    white_exempt       BOOLEAN NOT NULL DEFAULT FALSE,
    black_exempt       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS student_assignment_scores (
    id                 SERIAL PRIMARY KEY,
    student_id         INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    assignment_type_id INTEGER NOT NULL REFERENCES assignment_types(id) ON DELETE CASCADE,
    correct            INTEGER NOT NULL DEFAULT 0,
    incorrect          INTEGER NOT NULL DEFAULT 0,
    UNIQUE (student_id, assignment_type_id)
);

-- ── Seed: one "Homework" AssignmentType per classroom ─────────────────────────
-- Uses the most recent round's metric_mode/missing_policy for each classroom.
-- Classrooms with no rounds get defaults.

INSERT INTO assignment_types (classroom_id, name, metric_mode, missing_policy, created_at)
SELECT DISTINCT ON (r.classroom_id)
    r.classroom_id,
    'Homework',
    COALESCE(NULLIF(r.homework_metric_mode, ''), 'pct_correct'),
    COALESCE(NULLIF(r.missing_homework_policy, ''), 'zero'),
    NOW()
FROM rounds r
ORDER BY r.classroom_id, r.id DESC
ON CONFLICT DO NOTHING;

INSERT INTO assignment_types (classroom_id, name, metric_mode, missing_policy, created_at)
SELECT c.id, 'Homework', 'pct_correct', 'zero', NOW()
FROM classrooms c
WHERE NOT EXISTS (
    SELECT 1 FROM assignment_types at
    WHERE at.classroom_id = c.id AND at.name = 'Homework'
);

-- ── Seed: RoundAssignmentType for every existing round ────────────────────────

INSERT INTO round_assignment_types (round_id, assignment_type_id, weight, total_questions)
SELECT
    r.id,
    at.id,
    COALESCE(r.homework_weight, 30),
    COALESCE(r.homework_total_questions, 0)
FROM rounds r
JOIN assignment_types at
    ON at.classroom_id = r.classroom_id AND at.name = 'Homework'
WHERE NOT EXISTS (
    SELECT 1 FROM round_assignment_types rat
    WHERE rat.round_id = r.id AND rat.assignment_type_id = at.id
);

-- ── Seed: AssignmentEntry from homework_entries ──────────────────────────────

INSERT INTO assignment_entries (
    match_id, assignment_type_id,
    white_correct, white_incorrect, black_correct, black_incorrect,
    white_submitted, black_submitted, white_pct_wrong, black_pct_wrong,
    white_exempt, black_exempt
)
SELECT
    he.match_id,
    at.id,
    he.white_correct,   he.white_incorrect,
    he.black_correct,   he.black_incorrect,
    he.white_submitted, he.black_submitted,
    he.white_pct_wrong, he.black_pct_wrong,
    he.white_exempt,    he.black_exempt
FROM homework_entries he
JOIN matches m   ON m.id = he.match_id
JOIN rounds  r   ON r.id = m.round_id
JOIN assignment_types at
    ON at.classroom_id = r.classroom_id AND at.name = 'Homework'
WHERE NOT EXISTS (
    SELECT 1 FROM assignment_entries ae
    WHERE ae.match_id = he.match_id AND ae.assignment_type_id = at.id
);

-- ── Seed: StudentAssignmentScore from students.homework_correct/incorrect ─────

INSERT INTO student_assignment_scores (student_id, assignment_type_id, correct, incorrect)
SELECT
    s.id,
    at.id,
    s.homework_correct,
    s.homework_incorrect
FROM students s
JOIN assignment_types at
    ON at.classroom_id = s.classroom_id AND at.name = 'Homework'
WHERE (s.homework_correct > 0 OR s.homework_incorrect > 0)
ON CONFLICT (student_id, assignment_type_id) DO UPDATE
    SET correct   = EXCLUDED.correct,
        incorrect = EXCLUDED.incorrect;
