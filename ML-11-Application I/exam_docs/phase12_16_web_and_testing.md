# PHASE 12-16: Database, Web App, Analytics, Export, Testing — results

Scope decision (asked and confirmed): build the backend **and** real HTML
pages (not JSON-API-only), covering the plan's "Must have" + "Should have"
priority list (section 20 of again.txt). Deferred: full course/enrollment
management, PDF export, CLO web UI (CLOs work end-to-end at the API/DB/
Excel level, just no page to create them yet), Celery/Redis job queue,
production security hardening (HTTPS, rate limiting — PHASE 17 territory),
and anything requiring real users (STEP 16.3 inter-rater validation, 16.6
UAT).

## PHASE 12 — Database (STEP 12.1-12.5)

Extended the PHASE 10-11 schema (`exam_src/db/models.py`) with `users`, `students`
(a lightweight roster, not the plan's full students/enrollments pair —
see the module docstring for why), `clos`, and `question_stats`. Still not
the full plan schema: no `courses`/`enrollments` (exams aren't scoped to a
course here), and CLOs attach directly to an exam's `answer_keys` rows
instead of a separate `questions` table (`AnswerKey` already is "one row
per question").

**STEP 12.4 Alembic set up properly, not skipped**: `exam_migrations/` tracks
schema changes; `exam_src/db/session.py::init_db()` now runs
`alembic upgrade head` (idempotent) instead of a bare `create_all()`. Found
mid-work that the dev database already had un-migrated tables from PHASE
10-11 (predates Alembic); rather than hand-patch it, deleted the (fabricated,
gitignored, zero real data) dev `aegs.db` and generated one clean initial
migration against an empty database, verified by re-running the PHASE 10-11
demo end to end afterward.

**STEP 12.3 indices**: explicit index on `answer_sheets(exam_id, student_id)`;
the unique constraints on `extracted_answers(answer_sheet_id, question_no)`
and `results(exam_id, student_id)` already provide the other two indices
the plan asks for.

## PHASE 13 — Web application (STEP 13.1-13.5)

FastAPI app: `exam_src/api/main.py` (lifespan: runs migrations, seeds the admin
user, loads the CNN inference wrapper once at startup per STEP 13.2),
routers split by concern (`auth`, `exams`, `grading`, `results`, `utility`),
Pydantic schemas for every endpoint body (`exam_src/api/schemas.py`).

**The user's explicit ask** — a page to enter your own answer key — is
`exam_web/templates/exam_create.html`: create the exam, then a click-to-select
grid (one row per question, one button per choice, a score field, a
"cancel this question" checkbox) posts to `POST /api/exams/{id}/answer-key`
(`exam_src/api/routers/exams.py`), which replaces the exam's key. A "fill
example answers" button exists for quickly exercising the rest of the
pipeline without hand-clicking through 100 questions during testing.

Other pages: `dashboard.html`, `upload.html` (drag-and-drop,
background grading job with a polling progress bar), `review.html` (the
human review queue — shows the full uploaded sheet image, fill-ratio bars,
click or `1`-`9`/`0` keyboard shortcuts, auto-advances), `results.html`
(sortable/searchable score table, histogram via Chart.js, per-question
p/D table, Excel export button, "recompute scores" button).

**Shared access**: the local application intentionally has no login page or
authentication requirement, so multiple people can use it immediately.
API audit entries use the common identity `shared-user`; this mode does not
distinguish which person performed an edit.

**STEP 13.2 background jobs**: a single local worker thread with job state
and validated input paths persisted in SQLite. Queued/running jobs resume
from their last completed file after an application restart. This remains
a single-process desktop design rather than a distributed Celery cluster.
Model loaded once at startup, confirmed via `/api/health`.

**STEP 13.5 security implemented**: bcrypt password hashing, JWT httpOnly
cookies, role dependency (`require_admin`), upload file-size limit (50MB)
and magic-number content-type check (not just the extension), and a path-
traversal guard on the image-serving endpoint (resolved path must stay
under the uploads directory). **Not implemented**: HTTPS enforcement, rate
limiting, PDPA data-retention policy — deployment-time concerns (PHASE 17).

### End-to-end verification (not just "it imports")
Ran a real uvicorn server and drove the full flow with an HTTP client:
open dashboard → create exam → submit a 5-question answer key → upload a real scan
→ background-grade it → review queue (3 items, matching PHASE 9's own
BLANK+REVIEW count for that scan exactly) → submit one review → recompute
→ results (2 correct/3 incorrect against the test key — **hand-verified
against PHASE 9's independently-known answers for that scan, exact match**)
→ analytics → question-stats → Excel export (verified with openpyxl: all 4
sheets present, correct Thai names). Full walkthrough transcript kept in this session;
not re-saved as a repo artifact.

## PHASE 14 — Analytics (STEP 14.1-14.6)

`exam_src/analytics/item_analysis.py`: exam summary stats (mean/median/mode/SD/
quartiles/skewness/kurtosis via numpy+scipy), difficulty index p with the
plan's interpretation bands, discrimination index D (27% high/low split)
with its bands, distractor analysis (per-choice counts split by high/low
group), KR-20 reliability, and CLO analysis (mean % and pass rate per CLO).
STEP 14.7 (student ranking/percentile/z-score) and 14.8 (system
self-health report) were not built — straightforward additions on the
same data if wanted later.

All non-trivial formulas (p, D, KR-20) are unit-tested against hand-computed
values, not just "runs without crashing" (see PHASE 16 below).

## PHASE 15 — Export (STEP 15.1 only)

`exam_src/analytics/export_excel.py`: all 4 sheets from the plan (คะแนน,
สรุปสถิติ, วิเคราะห์ข้อ, CLO), conditional formatting on flagged questions
(negative D, or p in the extreme 0-5%/95-100% range), frozen header row,
autofilter. **PDF export (STEP 15.2), CSV/JSON export (15.3), and student
notification links (15.4) were not built.**

## PHASE 16 — Testing (STEP 16.1, 16.2, 16.5 done; 16.3/16.4/16.6 need real users/infra)

- **STEP 16.1 unit tests**: grading boundaries at exact edges (79.99% vs
  80.00%, `exam_tests/test_scoring.py`), p/D/KR-20 against hand-computed values
  (`exam_tests/test_analytics.py`) — 28 tests, all passing.
- **STEP 16.2 integration test with synthetic ground truth**
  (`exam_tests/test_integration_synthetic.py`): draws clean X marks at known
  positions on 10 synthetic sheets (1,000 questions total, including
  ~5% deliberately blank), runs them through the *real* PHASE 8+9 pipeline,
  and checks recovered answers against ground truth. **Result: 100.0%
  accuracy (1000/1000), including all 36 blank questions correctly
  detected as blank.** This validates pipeline logic on clean input; it is
  not evidence of real-world accuracy (no scan noise, no ambiguous
  handwriting — that needs STEP 16.3 with real graded sheets, not done).
- **STEP 16.5 robustness tests** (`exam_tests/test_robustness.py`), generated
  from a real scan:
  - **Handled correctly**: rotation up to 25° — perspective correction
    recovers the sheet and still extracts the correct student ID at every
    angle tested (12°/15°/20°/25°).
  - **A real bug was found and fixed**: a left-to-right shadow gradient
    dark enough to hide the left-side corner markers caused
    `select_four_corners()` to silently assign the *same* detected marker
    to two different corner slots (e.g. top-left == top-right). The
    resulting degenerate quadrilateral made `warpPerspective` output a
    uniform-gray, information-free image — while the code still reported
    `corners_found=True` and produced a garbage `"?????????????"` student
    ID with no error or warning. **Fixed** in `exam_src/vision/fiducial.py` by
    rejecting any corner assignment where two of the 4 slots resolve to
    (near) the same point (`MIN_CORNER_SEPARATION_PX`). Re-verified all 5
    real scans and all 4 rotation angles still pass after the fix (no
    regression), and the shadow case now correctly fails the quality gate
    instead of silently corrupting output.
  - Not tested: creases/folds, off-angle phone photos, 1-bit scans,
    erase-and-rewrite marks — would need either real degraded samples or
    more synthetic scenarios; not done in this pass.
- **STEP 16.3** (inter-rater agreement vs. real human TAs on 50-100 real
  sheets) — **cannot be done without real people and real graded sheets**;
  this project only has 5 sample scans and no real answer key.
- **STEP 16.4** (200 concurrent uploads, stress/memory test) — not run;
  would need a load-testing setup (e.g. locust) not currently in the repo.
- **STEP 16.6** (UAT with other teachers) — needs real users; not applicable
  to an automated pass.

## Files added this phase
`exam_src/db/models.py` (extended), `exam_migrations/` (new, Alembic), `exam_src/api/`
(new: `main.py`, `security.py`, `deps.py`, `schemas.py`,
`routers/{auth,exams,grading,results,utility}.py`), `exam_web/templates/*.html`,
`exam_web/static/css/style.css`, `exam_src/analytics/item_analysis.py`,
`exam_src/analytics/export_excel.py`, `exam_tests/test_scoring.py`,
`exam_tests/test_analytics.py`, `exam_tests/test_integration_synthetic.py`,
`exam_tests/test_robustness.py`.
