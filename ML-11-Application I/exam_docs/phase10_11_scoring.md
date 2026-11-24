# PHASE 10-11: Review Queue & Scoring — results

## Why a database shows up here (pulled forward from PHASE 12)
STEP 10.3 explicitly requires storing extraction results and review state
"ลงฐานข้อมูล" (into a database) — PHASE 10/11 can't be meaningfully built as
pure in-memory functions. `exam_src/db/models.py` implements the slice of the
PHASE 12 schema these two phases actually need: `exams`, `answer_keys`,
`answer_sheets`, `extracted_answers`, `reviewed_answers`, `audit_logs`,
`results`. Not the full PHASE 12 schema yet (no `courses`/`students`/
`enrollments`/`clos`/`users` tables) — those can be added when PHASE 12 is
tackled properly (migrations, indices, backup policy).

SQLite, per `config.yaml` (`database.driver`/`sqlite_path`) — file lives at
the project root (`aegs.db`), gitignored like other local data.

## PHASE 10 — Extraction storage & review queue

`exam_src/scoring/extraction_store.py`:
- `save_sheet_result()` (STEP 10.1): turns one `vision.pipeline.process_sheet()`
  result into an `AnswerSheet` row + 100 `ExtractedAnswer` rows.
- `get_review_queue()` (STEP 10.3): every `ExtractedAnswer` with
  `status != "OK"`, across sheets — what a review page would list.
- `submit_review()` (STEP 10.3/10.4): records a teacher's final answer as a
  `ReviewedAnswer` row and writes an `AuditLog` entry (who, when, old value,
  new value) in the same transaction. `ExtractedAnswer.final_answer` prefers
  the human review over the model's prediction automatically, so scoring
  code never has to know whether an answer was reviewed.

## PHASE 11 — Answer key & scoring

`exam_src/scoring/scoring.py`:
- `load_answer_key()` (STEP 11.1): one row per question; supports multiple
  correct choices per question and cancelled questions (full credit to
  everyone, added to both raw and max score so it doesn't shift the
  percentage).
- `score_sheet()` / `score_exam()` (STEP 11.2-11.3): compares
  `ExtractedAnswer.final_answer` (review-aware) against the key.
  Questions marked `REVIEW`/`MULTIPLE` retain the detector's strongest
  automatic choice and are scored immediately as provisional results while
  also being counted in `review_pending_count`. A later human review
  overrides that choice and triggers automatic rescoring.
  Optional negative marking (`negative_penalty`, default 0.25 of the
  question's score) verified: 29 correct + 1 cancelled + 68 incorrect (on
  the fabricated key) scores 30.0 without negative marking and 13.0 with
  it (29×1 − 68×0.25 + 1 = 13.0 — arithmetic checked by hand).
- `assign_grade_criterion()` / `assign_grade_norm()` (STEP 11.4): both grading
  tables from the plan implemented and spot-checked against known
  boundaries (e.g. exactly 80% -> A, 79.9% -> B+; T-score exactly 63 -> A).
- `sanity_check()` (STEP 11.5): negative/over-max scores, duplicate student
  IDs across sheets, unresolved student IDs, sheet-count mismatch.

## End-to-end validation (`exam_src/scoring/run_demo.py`)

Ran the full chain — PHASE 8/9 vision -> PHASE 10 storage -> a simulated
teacher review -> PHASE 11 scoring -> sanity check — against the 5 real
scans in `exam_input_files/`.

**⚠️ No real answer key exists for these sample sheets.** `run_demo.py`
uses a fabricated key (cycling ก→ข→ค→ง→ก...) purely to exercise the
arithmetic end-to-end. Every score in `exam_reports/phase10_11_demo.json` is a
synthetic plumbing check, not a real grade — do not read meaning into the
20-35% figures.

Verified:
- All 5 sheets stored correctly (`answer_sheets` + 500 `extracted_answers` rows).
- Review queue correctly listed 20 items (matches PHASE 9's own BLANK+REVIEW+MULTIPLE
  counts across the 5 sheets exactly).
- Submitting one review updated exactly that one student's
  `review_pending_count` and left everyone else unchanged.
- `sanity_check()` correctly flagged the same 3 sheets with unresolved
  student-ID digits that PHASE 9 already found — the two phases agree
  independently, which is the point of having the check.

## Not done in this pass
- The full PHASE 12 schema (courses, students, enrollments, CLOs, users) —
  only what PHASE 10-11 needed was built.
- Alembic migrations — `init_db()` uses `create_all()`, fine for a schema
  with no production data yet; swap over before the schema needs to change
  under real data.
- A real answer key for these sample sheets — none exists; scoring is only
  validated for correctness of arithmetic, not against real grades.
- The actual review web page (PHASE 13) — `submit_review()` is ready to be
  called from an endpoint, but no UI exists yet.
