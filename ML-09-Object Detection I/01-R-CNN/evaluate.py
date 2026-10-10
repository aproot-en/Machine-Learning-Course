


from pathlib import Path
import argparse
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "model.keras"

IMG_SIZE = 64
NUM_CLASSES = 10

CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]


def load_crops(split="test", limit=200):
    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Missing dataset folders: {images_dir} or {labels_dir}"
        )

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if limit > 0:
        image_files = image_files[:limit]

    xs, ys = [], []

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                continue

            if not 0 <= cls < NUM_CLASSES:
                continue

            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w, int((xc + bw / 2) * w))
            y2 = min(h, int((yc + bh / 2) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            xs.append(crop)
            ys.append(cls)

    if not xs:
        raise RuntimeError(
            f"No labeled crops in {images_dir}. "
            "Check YOLO label files."
        )

    return (
        np.asarray(xs, dtype=np.float32) / 255.0,
        np.asarray(ys, dtype=np.int32)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading dataset:", args.split)
    x, y = load_crops(args.split, args.limit)

    print("Total labeled crops:", len(y))
    print("Loading model:", MODEL_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)

    predictions = model.predict(
        x, batch_size=args.batch, verbose=1
    )

    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y, y_pred)

    print("\nEvaluation Results")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASSES,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
