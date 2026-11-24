"""Clean and split the Kaggle A-Z data, then save the processed arrays."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from exam_cnn.data.cleaning import clean_dataset
from exam_cnn.data.preprocessing import character_normalize_batch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_kaggle_az(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, header=None, dtype=np.uint8)
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy().reshape(-1, 28, 28)
    return X, y


def main():
    cfg = load_config()
    canvas_size = cfg["preprocessing"]["canvas_size"]
    target_edge = cfg["preprocessing"]["char_target_edge_px"]
    train_ratio = cfg["split"]["train"]
    val_ratio = cfg["split"]["validation"]
    test_ratio = cfg["split"]["test"]
    seed = cfg["split"]["random_state"]

    csv_path = PROJECT_ROOT / cfg["dataset"]["primary_path"]
    processed_dir = PROJECT_ROOT / cfg["paths"]["data_processed"]
    reports_dir = PROJECT_ROOT / cfg["paths"]["ml_reports"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {"steps": []}

    t0 = time.time()
    print(f"[1/5] Loading {csv_path.name} ...")
    X_raw, y_raw = load_kaggle_az(csv_path)
    print(f"      loaded {X_raw.shape[0]} rows in {time.time() - t0:.1f}s")
    report["steps"].append({"step": "load", "n_rows": int(X_raw.shape[0])})

    t0 = time.time()
    print("[2/5] Cleaning (dedupe + ink-ratio filter) ...")
    X_clean, y_clean, kept_index, clean_report = clean_dataset(X_raw, y_raw)
    print(
        f"      removed {clean_report['n_removed_total']} "
        f"({clean_report['n_duplicates_removed']} dup, {clean_report['n_bad_ink_removed']} bad-ink), "
        f"kept {clean_report['n_kept']} in {time.time() - t0:.1f}s"
    )
    report["steps"].append({"step": "clean", **clean_report})

    t0 = time.time()
    print(
        f"[3/5] Stratified split {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%} (seed={seed}) ..."
    )
    idx = np.arange(len(y_clean))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y_clean, train_size=train_ratio, stratify=y_clean, random_state=seed
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, train_size=relative_val, stratify=y_temp, random_state=seed
    )
    print(
        f"      train={len(idx_train)} val={len(idx_val)} test={len(idx_test)} in {time.time() - t0:.1f}s"
    )
    report["steps"].append(
        {
            "step": "split",
            "n_train": len(idx_train),
            "n_val": len(idx_val),
            "n_test": len(idx_test),
            "random_state": seed,
        }
    )

    t0 = time.time()
    print("[4/5] Character-normalizing (bbox crop -> resize -> center on mass) ...")
    X_train = character_normalize_batch(X_clean[idx_train], canvas_size, target_edge)
    X_val = character_normalize_batch(X_clean[idx_val], canvas_size, target_edge)
    X_test = character_normalize_batch(X_clean[idx_test], canvas_size, target_edge)
    print(f"      done in {time.time() - t0:.1f}s")

    class_weight_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weight_arr)}

    print("[5/5] Saving processed arrays + report ...")
    out_path = processed_dir / "az_dataset.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        # original-CSV row indices, for reproducibility (STEP 2.4)
        source_index_train=kept_index[idx_train],
        source_index_val=kept_index[idx_val],
        source_index_test=kept_index[idx_test],
    )
    print(f"      saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    with open(processed_dir / "class_weight.json", "w", encoding="utf-8") as f:
        json.dump(class_weight, f, indent=2)

    report["class_weight"] = class_weight
    report["output_file"] = str(out_path.relative_to(PROJECT_ROOT))
    with open(reports_dir / "phase2_preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
