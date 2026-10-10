


from pathlib import Path
import argparse
import random
import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUT / "model.keras"

CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]

IMG_SIZE = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    images_dir = DATASET / args.split / "images"
    labels_dir = DATASET / args.split / "labels"

    if args.image:
        image_path = Path(args.image)
    else:
        image_files = [
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            and (labels_dir / (p.stem + ".txt")).exists()
        ]

        if not image_files:
            raise RuntimeError("No labeled images found.")

        image_path = random.choice(image_files)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(str(image_path))

    label_path = labels_dir / (image_path.stem + ".txt")

    if not label_path.exists():
        raise FileNotFoundError(str(label_path))

    model = tf.keras.models.load_model(MODEL_PATH)

    h, w = image.shape[:2]
    crops = []
    boxes = []

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

        if not 0 <= cls < len(CLASSES):
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

        crops.append(crop)
        boxes.append((x1, y1, x2, y2))

    if not crops:
        raise RuntimeError("No valid labeled objects in image.")

    x = np.asarray(crops, dtype=np.float32) / 255.0
    predictions = model.predict(x, verbose=0)

    for box, pred in zip(boxes, predictions):
        x1, y1, x2, y2 = box

        cls = int(np.argmax(pred))
        confidence = float(np.max(pred))

        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            image,
            f"{CLASSES[cls]} {confidence:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

    OUT.mkdir(parents=True, exist_ok=True)
    save_path = OUT / "image_result.jpg"

    cv2.imwrite(str(save_path), image)

    print("Input:", image_path)
    print("Objects:", len(boxes))
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
