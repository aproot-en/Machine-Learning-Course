
from pathlib import Path
import argparse
import numpy as np
import cv2
import tensorflow as tf

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64
NUM_CLASSES = 10


def load_crops(split="train", limit=200):
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

        for line in label_path.read_text(encoding="utf-8").splitlines():
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
            f"No labeled crops found in {images_dir}. "
            "Check images and YOLO labels."
        )

    x = np.asarray(xs, dtype=np.float32) / 255.0
    y = np.asarray(ys, dtype=np.int32)

    return x, y


def make_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    x, y = load_crops("train", args.limit)
    print(f"Training {len(y)} labeled crops")

    model = make_model()

    model.fit(
        x, y,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=0.1 if len(y) >= 20 else 0,
        shuffle=True
    )

    model.save(OUT / "model.keras")
    print("Saved:", OUT / "model.keras")


if __name__ == "__main__":
    main()
