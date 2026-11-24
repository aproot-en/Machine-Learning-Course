# PHASE 7-9: Vision Pipeline — results

Built against the **real** answer sheet found in `exam_input_files/` (see
`exam_docs/answer_sheet_real_template.md`), not a newly-designed one.

## PHASE 7 — Template (STEP 7.1-7.5 superseded)

Instead of designing a new sheet, `exam_src/vision/build_template.py` **measures**
the real blank template (`exam_input_files/test-data-image-org/ORG-Test-image.png`)
and writes `exam_src/vision/templates/rmutt_answer_sheet_v1.json`:

- 4 fiducial corner circles, detected via contour + circularity/aspect
  filtering (`exam_src/vision/fiducial.py`).
- Student ID grid: 13 columns x 10 rows (digits 0-9), detected via
  bounding-box contour clustering.
- Answer grid: 100 questions as 5 groups x 20 rows x 4 choices (ก/ข/ค/ง),
  same clustering approach.
- The builder **asserts** exact expected counts (4 corners, 13x10 ID grid,
  20x20 answer grid) and fails loudly if the image changes and something
  no longer matches — it's meant to run once against a known-good blank
  template, not handle arbitrary inputs.

Verified visually: overlaying the extracted grid on both the blank template
and a real filled scan (after PHASE 8's perspective correction) shows every
box aligned correctly — see `exam_reports/debug/phase7_template_overlay_*.png`.

## PHASE 8 — Image processing (STEP 8.1-8.8)

`exam_src/vision/preprocess.py`:
- STEP 8.1 ingest: grayscale load (JPG/PNG/TIFF; PDF not implemented — no
  PDF sample was available to test against).
- STEP 8.2 enhance: CLAHE (uneven lighting) + light bilateral denoise.
- STEP 8.3 perspective correction: detect the 4 corner circles, warp so
  they land on the blank template's own corner positions — every processed
  scan then shares the same canonical geometry as the PHASE 7 template.
- STEP 8.4 deskew: not needed separately — perspective correction already
  handles it (per the plan's own note that this is usually redundant once
  8.3 succeeds).
- STEP 8.6 binarize: adaptive Gaussian threshold, inverted (white ink on
  black), matching the training data's convention.
- STEP 8.7 quality gate: resolution (config.yaml
  `answer_sheet.real_scan.min_short_side_px`, relaxed to 800px — see
  `exam_docs/answer_sheet_real_template.md`), blur (Laplacian variance),
  brightness, and corner-detection success. The QR check from the original
  plan is removed (the real sheet has none).
  - **Calibration note**: brightness range needed correcting from an
    initial guess of (30, 225) to (100, 253) — a sparse-ink white answer
    sheet legitimately averages ~230-240 mean brightness (mostly blank
    page), which the original guess would have wrongly rejected on every
    single real sample.

All 5 real scans in `exam_input_files/test-data-image-org/` pass the quality gate
and get correctly perspective-corrected (verified against 2 different
source resolutions: 850x1200 and a resized 819x1161 copy).

## PHASE 9 — Answer detection (STEP 9.1-9.3; 9.4/9.5 not applicable)

`exam_src/vision/answer_detection.py`:
- STEP 9.1 Student ID: **Path B only** (OMR bubble grid) — the sheet has no
  QR code (Path A) and no handwritten digit boxes (Path C).
- STEP 9.2: box positions come from the PHASE 7 template, applied directly
  to the perspective-corrected image — never re-detected per scan.
- STEP 9.3: fill-ratio decision, adapted from "circle" to "X-mark-in-square-box"
  per the real sheet's design. Thresholds were **calibrated empirically**
  against the 5 real scans (fill-ratio histograms are clearly bimodal:
  unmarked boxes cluster 0.0-0.15, marked boxes cluster 0.28-0.57):
  - `BLANK_THRESHOLD = 0.12` — confidently no ink.
  - `AUTO_ACCEPT_THRESHOLD = 0.30` — a uniquely highest choice at or above
    30% is accepted immediately and does not require manual review.
  - Values below 30% -> `REVIEW`, not `BLANK` — added after finding a real
    example (scan 4, Q1) where a faint partial slash mark (ratio 0.205)
    should be routed to a human, not silently discarded as "no answer".
  - `SEPARATION_MARGIN = 0.10` for `MULTIPLE` detection.
- STEP 9.4 (CNN letter-in-box mode) and 9.5 (cross-check) do not apply —
  the sheet has no handwritten-letter answer zone.

### Correctness checks performed
1. **Cross-referenced against independently-read handwritten text — twice,
   because the first calibration was still wrong.** The sheet has a second,
   independent encoding of most of the student ID: a handwritten (blue
   pen) code in the "กระดาษคำตอบวิชา" box (`crop_subject_code_box()`,
   `subject_code_box` in template.json). Cropped and read that box by eye
   for **all 5** real scans (not just a couple), giving real ground truth:
   `1493314x4541x`, `123142463531x`, `124137531x31x`, `1315412583419`,
   `2351134213421`.

   *Why the OMR grid needed this second signal at all*: every ID bubble
   circle has a printed numeral (0-9) inside it, so even a fully unbubbled
   circle has non-trivial "ink" from that printed glyph alone (~0.2-0.5
   fill ratio), while a genuinely bubbled circle reaches ~0.55-0.98. A
   left-blank column is this institution's own convention for a literal
   `x` in the ID.

   *First calibration (`ID_BLANK_MAX=0.50`) was still wrong*: cross-checking
   against all 5 handwritten codes (not just the 2 checked earlier) caught
   a real bug -- scan 2's column 7 has a **lightly bubbled "4"** at top
   ratio 0.455, which 0.50 confidently misread as blank (`x`). Confirmed
   true blanks across all 5 scans top out at 0.354, so `ID_BLANK_MAX` was
   lowered to **0.40**, comfortably between the two clusters, and
   re-verified against all 5 ground-truth strings again -- exact match on
   every one.

   *The gap check earns its keep*: separately, every "?" `REVIEW` position
   across scans 4-5 (6 columns total) was checked against the handwritten
   ground truth, and in **every single case** the fill-ratio top candidate
   -- the one `ID_GAP_MIN` refused to trust outright -- was in fact the
   correct digit. So `detect_student_id()` now also returns
   `best_guess_id`: the same string with `?` positions filled in by that
   top candidate, surfaced in the review page as a pre-filled suggestion
   next to the handwritten-code image, rather than a bare "?" the teacher
   has to solve from nothing. `student_id` itself still shows `?` and
   stays `REVIEW` -- the guess is a hint, not an auto-applied answer.
2. Two independent reading methods (OMR bubble fill vs. eyeballing the
   handwritten text) agreeing digit-for-digit on every confidently-resolved
   position across all 5 scans is strong evidence the detector is correct,
   not coincidentally plausible.
3. **A real numbering bug was caught and fixed during this work**: the
   first implementation of `detect_answers` assigned question numbers by
   iterating rows-then-groups (`q=1,2,3,4,5` for row 0 across all 5
   groups), rather than groups-then-rows as the sheet actually lays out
   questions (group 0 = Q1-20, group 1 = Q21-40, ...). Caught by comparing
   output against a manual visual reading of row 1 of a real scan
   (`ก,ก,ก,ค,ก` across the 5 groups) — the fill-ratio computation itself
   was correct, only the label was wrong. Fixed before this was used
   anywhere downstream.
4. **Visual spot-check of a flagged case**: cropped and viewed the exact
   box behind scan 4's Q1 `REVIEW` — it is a genuine faint/partial mark,
   not a detection artifact.

### Results across the 5 real scans (`exam_reports/phase9_extraction_sample.json`)
| scan | student_id | best_guess_id | matches ground truth? | OK | REVIEW | BLANK | MULTIPLE |
|---|---|---|---|---|---|---|---|
| CCF_000001 | 1493314x4541x | 1493314x4541x | yes | 97 | 2 | 1 | 0 |
| CCF_000002 | 123142?63531x | 123142463531x | yes | 100 | - | - | - |
| CCF_000003 | 124137531x31x | 124137531x31x | yes | 96 | 1 | 2 | 1 |
| CCF_000004 | 1?15412583419 | 1315412583419 | yes | 92 | 3 | 2 | 3 |
| CCF_000005 | 2?5???4?13421 | 2351134213421 | yes | 95 | 2 | 3 | 0 |

`best_guess_id` matches the independently-read handwritten ground truth on
**all 5/5 scans, every digit** -- including the positions `student_id`
correctly declines to assert (`?`). 96% of all 500 answer-boxes-across-5-sheets
resolved confidently (`OK`); the rest correctly routed to human review per
the plan's human-in-the-loop design (PHASE 10 STEP 10.3), not silently
guessed.

**Existing live data was backfilled**, not just newly-processed scans: the
already-graded sheets in the running app's database were recomputed
in place with the fixed thresholds (student_id / student_id_best_guess /
student_id_status / student_id_confidence only -- answers and scores
untouched), and any Result row left stale by a student_id change was
dropped so "คำนวณคะแนนใหม่" regenerates it cleanly under the corrected ID.

Student ID `x` (confidently blank -- see the ID-grid fix above) appears in
3/5 scans (1, 2, 3), always including the very last (13th) column; whether
that column is conventionally left blank by design (e.g. a check digit) or
happens to be `x` in these particular students' codes isn't established
here. Scans 4 and 5 have several genuinely ambiguous columns (`?`,
`REVIEW`) where pencil pressure was too light relative to that scan's own
printed-numeral baseline to confidently call blank-vs-marked -- correctly
left for a human rather than guessed either way.

## Not done in this pass
- PDF ingestion (STEP 8.1) — no PDF sample was available to build/test against.
- STEP 9.4/9.5 (CNN letter mode, cross-check) — not applicable to this sheet.
- Calibrating against more than 5 samples — real-world threshold robustness
  beyond these specific scans is unverified.
- PHASE 10's review queue, database storage, and audit trail — `exam_src/vision/pipeline.py`
  produces the per-sheet result in memory; wiring it to a queue/DB is PHASE 10 work.
