# Real Answer Sheet Template (found in `exam_input_files/`)

Supersedes the "design from scratch" assumptions in `again.txt` PHASE 7. This
sheet already exists and is in production use — build the vision pipeline to
read *this*, not a new design, unless the user later asks for a new sheet.

- Source: `exam_input_files/test-data-image-org/ORG-Test-image.png` (blank template),
  cross-checked against 5 filled real scans (`CCF_000001-5.png`) and 3 debug
  overlays (`exam_incorrect_results/03-05.png`) showing a prior detection attempt's
  region-of-interest boxes.
- Issuer: คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี (RMUTT Engineering Faculty).

## Header
- Top-left: faculty/university name (Thai text, 2 lines).
- Top-right: "คะแนนที่ได้" (score) — a blank box, filled by hand/grader after scoring.
- One solid black filled circle at each of the 4 corners — the alignment/fiducial
  markers. **Circles, not squares or ArUco markers** as `again.txt` STEP 7.1.1 assumed.
- **No QR code or barcode anywhere on the sheet.**

## Student ID block
- Label: "กรอกตัวเลขรหัสนักศึกษาลงในช่องว่าง และระบายทึบในวงกลมตามหลักเข" (fill digit
  boxes, then bubble the matching circle).
- 13 digit columns, each 0–9 bubbled as circles (OMR), with a "−" (dash) printed
  between column 12 and 13 — i.e. a 12-digit ID + 1 check/suffix digit.
- This matches the plan's **Path A (recommended)**: bubble digits, not handwritten
  or QR. Path A in STEP 9.1 is the one to implement; Path B (handwritten digits +
  CNN) and QR-based ID are not applicable to this sheet.

## Exam/subject metadata block (handwritten, not machine-encoded)
- "กระดาษคำตอบวิชา" — a boxed field with a handwritten alphanumeric subject/exam
  code in blue pen, e.g. `1493314x4541-x` (digits + literal "x" separators seen
  across samples — format not fully confirmed, looks like `xxxxxxxxxxxxxx`-ish
  code with dashes; get the actual coding scheme from the teacher/registrar
  before building parsing logic).
- "ชื่อ-นามสกุล .........." (name), "ภาควิชา .........." (department),
  "สอบวันที่ .... เวลา ...." (exam date/time) — blank ruled lines, filled by hand
  in cursive Thai. **Out of scope for the trained A-Z Latin-letter CNN** (PHASE
  1-2 work) — these fields are not part of the automated grading path; treat as
  metadata to store as scanned images / manual entry, not OCR targets for now.

## Answer grid — this is the part that changes the CNN's role
- **100 questions, 5 columns × 20 rows** (not 4×25 as `again.txt` STEP 7.2 assumed).
- **4 choices per question: ก / ข / ค / ง** (not A–E, not 5 choices).
- Each choice is its own small square box with the Thai letter pre-printed
  faintly inside; the student marks their chosen box with an **X (กากบาท)** —
  a separate box per choice, not one box to write a letter into.
- The sheet itself prints calibration examples:
  - "ตัวอย่างกากบาทที่ถูกต้อง" (valid X examples): clean ✕ marks.
  - "ตัวอย่างกากบาทไม่ถูกต้อง" (invalid marks): a faint/incomplete X, an
    over-filled box (⊠-like), and a checkmark/pen-flick (✓-like) — these are
    the sheet's own examples of what should NOT count as a valid answer.
    Real scans confirm students actually produce all of these variants
    (slashes, checkmarks, light marks, double-marks) — calibrate the fill/
    shape threshold against these, not just a clean X.
  - "หมายเหตุ ถ้าไม่ต้องการคำตอบให้ระบายเต็มช่อง" — a fully-shaded (not X'd) box
    means "no answer" / voided, distinct from an unmarked box.

### Implication for the CNN work already done (PHASE 1–3)
The Kaggle A-Z model is **not** the primary mechanism for reading this sheet's
100 answers — that's fundamentally an OMR fill/mark-detection problem per box
(4 boxes/question × 100 questions), much closer to `again.txt` STEP 9.3 (OMR)
than STEP 9.4 (CNN letter classification). `again.txt`'s own overview point
(3) already anticipated an "X mark" detection need — this confirms it's the
actual primary path, and STEP 9.4's handwritten-letter-in-one-box scenario
does not apply to this sheet.
The 26-class A-Z CNN work from PHASE 1–3 is not wasted — it stays useful as a
general handwritten-character reader (e.g. if a future sheet design does use
handwritten letters, or for OCR-ing the subject-code field if that turns out
to contain A-Z characters) — just not required to grade *this* sheet's 100
questions. Treat it as a secondary/optional capability, not a blocker.

## Resolution / scan quality — fails the plan's own quality gate
| Source | Size (px) | Short side | DPI (metadata) |
|---|---|---|---|
| `again.txt` STEP 8.1 minimum | — | **≥ 1200 px** | assumes 300 |
| `exam_input_files/*.png` | 819×1161 | 819 | ~96–100 |
| `exam_input_files/test-data-image-org/*.png` | 850×1200 | 850 | ~96–100 |
| `exam_incorrect_results/*.png` | 1152×1200 | 1152 | ~96 |

None of the available real samples meet the plan's own minimum-resolution
quality gate. Two options going forward (not decided yet):
1. Re-scan future samples at 300 DPI (real A4 at 300 DPI ≈ 2480×3508) so the
   plan's fine-feature assumptions (sub-mm marker detection, crisp box edges)
   hold.
2. Lower PHASE 8's quality-gate threshold to match what the actual scanner in
   use produces (~850–1200 px short side, ~96–100 DPI), and re-tune anything
   that assumed 300 DPI (e.g. fiducial marker size in pixels, box crop padding).
This project will proceed assuming option 2 (work with what the real scanner
produces) until told otherwise, since re-scanning historical samples isn't
possible and the production scanner is unlikely to change.

## Evidence of a prior detection attempt
`exam_incorrect_results/03-05.png` are the *same* scans as `CCF_000003-5.png` with
red rectangles overlaid (around each corner marker, the full student-ID grid
per-column, and top/bottom banner strips) plus small green tick marks next to
specific answer rows (e.g. row 1, row 11/31, row 81 in different samples).
This looks like debug output from an earlier (not-this-project) detection
script that mis-detected something at those green-marked rows — no code for
that attempt was found in this repo. If the user has that old script
somewhere, it may be worth reviewing before reimplementing PHASE 8/9 from
scratch, since it apparently already solved corner/grid detection to some
extent.
