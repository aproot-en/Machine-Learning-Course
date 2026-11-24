"""Evaluate CNN15 accuracy, confidence, errors, and inference speed."""

import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from exam_cnn.data.preprocessing import character_normalize_batch, normalize_pixels
from exam_cnn.models.cnn15 import CNN15

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "exam_cnn/datasets/processed/az_dataset.npz"
CHECKPOINT_PATH = PROJECT_ROOT / "exam_cnn/models_export/cnn15_best.pt"
EMNIST_DIR = PROJECT_ROOT / "exam_cnn/datasets/raw/emnist/EMNIST/raw"
REPORTS_DIR = PROJECT_ROOT / "exam_cnn/reports"
DEBUG_DIR = REPORTS_DIR / "debug"
DOCS_PATH = PROJECT_ROOT / "exam_cnn/docs/phase5_evaluation.md"

BATCH_SIZE = 256
CONFIDENCE_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.99]
TARGET_ACCURACY_AT_THRESHOLD = 0.995
N_ERROR_EXAMPLES = 100
SPEED_WARMUP_BATCHES = 5
SPEED_MEASURE_BATCHES = 50
QUESTIONS_PER_SHEET = 100
STUDENTS_PER_ROOM = 60

LETTERS = [chr(65 + i) for i in range(26)]


def read_idx(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        return np.frombuffer(f.read(), dtype=np.uint8)


@torch.no_grad()
def predict_all(model, X: np.ndarray, device: str, batch_size: int = BATCH_SIZE):
    """Return predicted labels, confidences, and class probabilities for an image batch."""
    model.eval()
    Xn = character_normalize_batch(X)
    preds, confs, all_probs = [], [], []
    for i in range(0, len(Xn), batch_size):
        batch = normalize_pixels(Xn[i : i + batch_size])[:, None, :, :]
        xb = torch.from_numpy(batch).to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs), np.concatenate(all_probs)


def report_test_metrics(y_true, y_pred, report_lines):
    acc = (y_true == y_pred).mean()
    report = classification_report(
        y_true, y_pred, target_names=LETTERS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(26)))
    cm_norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    np.savetxt(REPORTS_DIR / "confusion_matrix_test.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(
        REPORTS_DIR / "confusion_matrix_test_normalized.csv", cm_norm, fmt="%.4f", delimiter=","
    )

    report_lines.append("## Test Metrics\n")
    report_lines.append(f"- Overall accuracy: **{acc:.4f}**")
    report_lines.append(f"- Macro F1: **{report['macro avg']['f1-score']:.4f}**")
    report_lines.append(f"- Weighted F1: **{report['weighted avg']['f1-score']:.4f}**\n")

    worst = sorted(
        ((LETTERS[i], report[LETTERS[i]]["f1-score"]) for i in range(26)), key=lambda t: t[1]
    )[:5]
    report_lines.append(f"- Lowest-F1 classes: {', '.join(f'{l}={f:.3f}' for l, f in worst)}\n")

    report_lines.append("## Confusion Matrix\n")
    report_lines.append(
        "Saved to `exam_cnn/reports/confusion_matrix_test.csv` (raw counts) and "
        "`exam_cnn/reports/confusion_matrix_test_normalized.csv` (row-normalized).\n"
    )
    off_diag = cm_norm.copy()
    np.fill_diagonal(off_diag, 0)
    top_confusions = []
    for i in range(26):
        j = off_diag[i].argmax()
        if off_diag[i, j] > 0.02:
            top_confusions.append(f"{LETTERS[i]}->{LETTERS[j]} ({off_diag[i, j]:.1%})")
    report_lines.append(
        f"- Most-confused pairs (>2% of a class's predictions): {', '.join(top_confusions) or 'none'}\n"
    )
    return acc


def report_confidence(y_true, y_pred, confs, report_lines):
    correct = y_true == y_pred
    report_lines.append("## Confidence\n")
    report_lines.append("| threshold | coverage | accuracy@threshold |")
    report_lines.append("|---|---|---|")
    chosen = None
    for t in CONFIDENCE_THRESHOLDS:
        mask = confs >= t
        coverage = mask.mean()
        acc_at_t = correct[mask].mean() if mask.any() else float("nan")
        report_lines.append(f"| {t:.2f} | {coverage:.1%} | {acc_at_t:.4f} |")
        if acc_at_t >= TARGET_ACCURACY_AT_THRESHOLD and chosen is None:
            chosen = t
    report_lines.append("")
    if chosen is not None:
        report_lines.append(
            f"- Lowest threshold reaching >= {TARGET_ACCURACY_AT_THRESHOLD:.1%} accuracy: **{chosen}**\n"
        )
    else:
        report_lines.append(
            f"- No tested threshold reached {TARGET_ACCURACY_AT_THRESHOLD:.1%} accuracy "
            f"-- consider raising CONFIDENCE_THRESHOLDS or accept a lower target.\n"
        )
    np.savez(REPORTS_DIR / "confidence_analysis.npz", confs=confs, correct=correct)
    return chosen


def report_errors(X_test_raw, y_true, y_pred, confs, report_lines):
    wrong_idx = np.nonzero(y_true != y_pred)[0]
    report_lines.append("## Errors\n")
    report_lines.append(
        f"- {len(wrong_idx)} / {len(y_true)} test images misclassified ({len(wrong_idx) / len(y_true):.2%})."
    )
    if len(wrong_idx) == 0:
        report_lines.append("")
        return
    sample = wrong_idx[np.argsort(-confs[wrong_idx])][
        :N_ERROR_EXAMPLES
    ]  # most confidently wrong first

    import math

    n = len(sample)
    cols = 10
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * 30, cols * 30), dtype=np.uint8)
    for k, idx in enumerate(sample):
        r, c = divmod(k, cols)
        canvas[r * 30 : r * 30 + 28, c * 30 : c * 30 + 28] = X_test_raw[idx]
    from PIL import Image

    Image.fromarray(canvas).save(DEBUG_DIR / "phase5_misclassified_top100.png")

    report_lines.append(
        f"- Saved the {n} most-confidently-wrong examples to "
        f"`exam_cnn/reports/debug/phase5_misclassified_top100.png` (10/row, true label order not shown -- "
        f"cross-reference indices in `exam_cnn/reports/error_analysis.json`)."
    )
    with open(REPORTS_DIR / "error_analysis.json", "w") as f:
        json.dump(
            {
                "n_wrong": int(len(wrong_idx)),
                "n_total": int(len(y_true)),
                "sample_indices": sample.tolist(),
                "sample_true": [LETTERS[y_true[i]] for i in sample],
                "sample_pred": [LETTERS[y_pred[i]] for i in sample],
                "sample_confidence": confs[sample].tolist(),
            },
            f,
            indent=2,
        )
    report_lines.append(
        "- Automated categorization of *why* each is wrong (illegible / mislabeled / "
        "preprocessing artifact) requires human review of the saved montage -- not done here.\n"
    )


def evaluate_emnist(model, device, report_lines):
    report_lines.append("## EMNIST Letters Test\n")
    images_path = EMNIST_DIR / "emnist-letters-test-images-idx3-ubyte"
    labels_path = EMNIST_DIR / "emnist-letters-test-labels-idx1-ubyte"
    if not images_path.exists():
        report_lines.append("- EMNIST files not found, skipped.\n")
        return

    X_emnist = read_idx(images_path)
    y_emnist = read_idx(labels_path)
    # EMNIST images are stored transposed relative to normal reading orientation
    # (verified empirically: raw bytes read row-major render rotated; img.T is correct).
    X_emnist = np.transpose(X_emnist, (0, 2, 1))
    # EMNIST 'letters' labels are 1-26 for a-z with upper/lower merged; map to
    # our 0-25 A-Z indexing (label L -> letter index L-1).
    y_emnist_mapped = y_emnist.astype(np.int64) - 1

    pred, conf, _ = predict_all(model, X_emnist, device)
    acc = (pred == y_emnist_mapped).mean()
    report_lines.append(
        f"- EMNIST Letters test accuracy (cross-dataset, orientation-corrected): **{acc:.4f}** "
        f"on {len(y_emnist)} images."
    )
    report_lines.append(
        "- A large drop vs. the Kaggle test accuracy above would indicate overfitting to "
        "Kaggle-specific handwriting style rather than general A-Z recognition.\n"
    )
    return acc


def benchmark_inference(model, device, report_lines):
    report_lines.append("## Inference Speed\n")
    dummy = torch.randn(1, 1, 28, 28, device=device)
    batch = torch.randn(BATCH_SIZE, 1, 28, 28, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(SPEED_WARMUP_BATCHES):
            model(dummy)
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(SPEED_MEASURE_BATCHES):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        per_image_ms = (time.time() - t0) / SPEED_MEASURE_BATCHES * 1000

        t0 = time.time()
        for _ in range(SPEED_MEASURE_BATCHES):
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        per_batch_ms = (time.time() - t0) / SPEED_MEASURE_BATCHES * 1000

    report_lines.append(f"- Single-image inference: {per_image_ms:.2f} ms/image")
    report_lines.append(
        f"- Batch-of-{BATCH_SIZE} inference: {per_batch_ms:.2f} ms/batch "
        f"({per_batch_ms / BATCH_SIZE:.3f} ms/image effective)"
    )
    sheet_s = QUESTIONS_PER_SHEET * (per_batch_ms / BATCH_SIZE) / 1000
    room_s = sheet_s * STUDENTS_PER_ROOM
    report_lines.append(
        f"- Estimated time for one {QUESTIONS_PER_SHEET}-question sheet (batched): {sheet_s * 1000:.1f} ms"
    )
    report_lines.append(
        f"- Estimated time for a {STUDENTS_PER_ROOM}-student room: {room_s:.2f} s "
        f"(model inference only -- excludes image processing/OMR, and this model is not on "
        f"the primary grading path for the real X-mark answer sheet; see "
        f"exam_docs/answer_sheet_real_template.md)\n"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    model = CNN15().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    d = np.load(DATA_PATH)
    X_test_raw, y_test = d["X_test"], d["y_test"]

    print("Predicting on test set ...")
    # X_test is already character-normalized (PHASE 2 output) -- normalizing
    # again is a no-op for already-centered images, kept for a single code path.
    y_pred, confs, _ = predict_all(model, X_test_raw, device)

    report_lines = [
        "# Model Evaluation\n",
        f"Checkpoint: epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}\n",
    ]

    test_acc = report_test_metrics(y_test, y_pred, report_lines)
    chosen_threshold = report_confidence(y_test, y_pred, confs, report_lines)
    report_errors(X_test_raw, y_test, y_pred, confs, report_lines)
    emnist_acc = evaluate_emnist(model, device, report_lines)
    benchmark_inference(model, device, report_lines)

    report_lines.append("## Real Student Handwriting\n")
    report_lines.append(
        "- Skipped: STEP 1.6 (collecting real student handwriting) was never run, "
        "so no held-out real-world test set exists. This means the accuracy numbers "
        "above are Kaggle/EMNIST accuracy, not proof of real-world performance.\n"
    )

    DOCS_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Test accuracy: {test_acc:.4f}")
    print(
        f"EMNIST cross-dataset accuracy: {emnist_acc:.4f}"
        if emnist_acc is not None
        else "EMNIST: skipped"
    )
    print(f"Chosen confidence threshold: {chosen_threshold}")
    print(f"Full report written to {DOCS_PATH}")

    with open(REPORTS_DIR / "phase5_summary.json", "w") as f:
        json.dump(
            {
                "test_accuracy": float(test_acc),
                "emnist_cross_dataset_accuracy": float(emnist_acc)
                if emnist_acc is not None
                else None,
                "chosen_confidence_threshold": chosen_threshold,
                "checkpoint_epoch": ckpt["epoch"],
                "checkpoint_val_acc": ckpt["val_acc"],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
