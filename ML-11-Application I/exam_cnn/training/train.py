"""Train CNN15, save the best checkpoint, and log the results."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from exam_cnn.data.dataset import make_dataloader
from exam_cnn.models.cnn15 import (
    BATCH_SIZE,
    CNN15,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LABEL_SMOOTHING,
    NUM_CLASSES,
    build_loss,
    build_optimizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "exam_cnn/datasets/processed/az_dataset.npz"
CLASS_WEIGHT_PATH = PROJECT_ROOT / "exam_cnn/datasets/processed/class_weight.json"
CHECKPOINT_PATH = PROJECT_ROOT / "exam_cnn/models_export/cnn15_best.pt"
CSV_LOG_PATH = PROJECT_ROOT / "exam_cnn/reports/training_log.csv"
TENSORBOARD_DIR = PROJECT_ROOT / "exam_cnn/reports/tensorboard"
EXPERIMENTS_LOG_PATH = PROJECT_ROOT / "exam_cnn/reports/experiments.csv"

SEED = 42
NUM_WORKERS = 0  # 0 is safest on Windows; raise if I/O-bound
USE_CLASS_WEIGHT = True

# ReduceLROnPlateau
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 5
MIN_LR = 1e-6

# smoke test
SMOKE_TEST_SUBSET = 5000
SMOKE_TEST_EPOCHS = 2
OVERFIT_N = 100
OVERFIT_MAX_EPOCHS = 200


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data():
    with np.load(DATA_PATH) as data:
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["y_test"],
        )


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += xb.size(0)
    return total_loss / total_n, total_correct / total_n


def smoke_test(device):
    """Check a short training run and confirm the model can fit a small sample."""
    print("=== Training checks ===")
    X_train, y_train, X_val, y_val, *_ = load_data()

    print(f"[smoke] {SMOKE_TEST_EPOCHS} epochs on {SMOKE_TEST_SUBSET} images ...")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_train), SMOKE_TEST_SUBSET, replace=False)
    loader = make_dataloader(
        X_train[idx],
        y_train[idx],
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )
    val_loader = make_dataloader(
        X_val[:1000],
        y_val[:1000],
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )

    model = CNN15().to(device)
    loss_fn = build_loss()
    optimizer = build_optimizer(model)
    for epoch in range(SMOKE_TEST_EPOCHS):
        tr_loss, tr_acc = run_epoch(model, loader, loss_fn, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        print(
            f"  epoch {epoch + 1}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
    print("[smoke] pipeline ran without error.")

    print(f"[overfit] trying to hit 100% train accuracy on {OVERFIT_N} images ...")
    idx = rng.choice(len(X_train), OVERFIT_N, replace=False)
    loader = make_dataloader(
        X_train[idx], y_train[idx], train=False, batch_size=OVERFIT_N, num_workers=0, seed=SEED
    )
    model = CNN15().to(device)
    loss_fn = build_loss()
    optimizer = build_optimizer(model)
    acc = 0.0
    for epoch in range(OVERFIT_MAX_EPOCHS):
        _, acc = run_epoch(model, loader, loss_fn, optimizer, device, train=True)
        if acc >= 1.0:
            break
    print(
        f"[overfit] reached train_acc={acc:.4f} after {epoch + 1} epochs "
        f"({'PASS' if acc >= 1.0 else 'FAIL -- check the pipeline for bugs'})"
    )
    print("=== smoke test done ===\n")
    return acc >= 1.0


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ok = smoke_test(device)
    if not ok:
        print("Smoke test failed to overfit 100 images -- aborting real training.")
        return

    print("=== Training CNN15 ===")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    train_loader = make_dataloader(
        X_train, y_train, train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=SEED
    )
    val_loader = make_dataloader(
        X_val, y_val, train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=SEED
    )

    model = CNN15().to(device)

    class_weight_tensor = None
    if USE_CLASS_WEIGHT:
        with open(CLASS_WEIGHT_PATH) as f:
            cw = json.load(f)
        class_weight_tensor = torch.tensor(
            [cw[str(i)] for i in range(NUM_CLASSES)], dtype=torch.float32, device=device
        )
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=LABEL_SMOOTHING)

    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR))

    best_val_acc = -1.0
    epochs_without_improvement = 0

    with open(CSV_LOG_PATH, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]
        )

        for epoch in range(EPOCHS):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(
                model, val_loader, loss_fn, optimizer, device, train=False
            )
            scheduler.step(val_loss)
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"epoch {epoch + 1}/{EPOCHS}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.2e} ({dt:.1f}s)"
            )
            csv_writer.writerow([epoch + 1, tr_loss, tr_acc, val_loss, val_acc, lr, dt])
            f.flush()
            writer.add_scalars("loss", {"train": tr_loss, "val": val_loss}, epoch + 1)
            writer.add_scalars("accuracy", {"train": tr_acc, "val": val_acc}, epoch + 1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    CHECKPOINT_PATH,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping at epoch {epoch + 1} (no val_acc improvement for {EARLY_STOPPING_PATIENCE} epochs)"
                    )
                    break

    writer.close()
    print(f"Best val_acc: {best_val_acc:.4f} -> saved to {CHECKPOINT_PATH}")

    # restore_best_weights=True -- reload best checkpoint before final report
    best = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(best["model_state_dict"])

    test_loader = make_dataloader(
        X_test, y_test, train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=SEED
    )
    test_loss, test_acc = run_epoch(model, test_loader, loss_fn, optimizer, device, train=False)
    print(f"Final test_acc (best checkpoint): {test_acc:.4f}")

    # fine-tuning with real student handwriting -- not done.
    # (collecting real student handwriting samples) was never run,
    # so there is no fine-tuning set available. Skipped.

    # experiment tracking
    EXPERIMENTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not EXPERIMENTS_LOG_PATH.exists()
    with open(EXPERIMENTS_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(
                [
                    "run_id",
                    "architecture",
                    "lr",
                    "batch_size",
                    "class_weight",
                    "seed",
                    "best_epoch",
                    "best_val_acc",
                    "test_acc",
                    "notes",
                ]
            )
        w.writerow(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                "cnn15",
                optimizer.defaults.get("lr"),
                BATCH_SIZE,
                USE_CLASS_WEIGHT,
                SEED,
                best["epoch"],
                best_val_acc,
                test_acc,
                "PHASE 4 initial run; no fine-tuning set (STEP 1.6 not collected)",
            ]
        )

    print("Training artifacts:")
    print(f"  checkpoint: {CHECKPOINT_PATH}")
    print(f"  csv log:    {CSV_LOG_PATH}")
    print(f"  tensorboard: {TENSORBOARD_DIR}")
    print(f"  experiments: {EXPERIMENTS_LOG_PATH}")


if __name__ == "__main__":
    main()
