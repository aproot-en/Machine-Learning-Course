import os

import numpy as np
import pandas as pd

IMG_SIZE = 48


def load_data(csv_path, others_dir):
    """Read age_gender.csv. Returns (pixels, meta)."""

    pixels_path = os.path.join(others_dir, "pixels.npy")
    meta_path = os.path.join(others_dir, "meta.csv")

    # Reuse the parsed copy if it exists
    if os.path.isfile(pixels_path) and os.path.isfile(meta_path):
        print(f"Reused: {others_dir}")
        return np.load(pixels_path), pd.read_csv(meta_path)

    # Read in chunks. The pixels column holds 2,304 numbers as one string,
    # so reading the whole file at once needs several GB of RAM.
    pixel_list = []
    meta_list = []

    for chunk in pd.read_csv(csv_path, chunksize=2000):
        pixel_list.append(np.array(
            [np.array(s.split(), dtype=np.uint8) for s in chunk["pixels"]]
        ))
        meta_list.append(chunk[["age", "ethnicity", "gender"]])

    pixels = np.concatenate(pixel_list)
    meta = pd.concat(meta_list, ignore_index=True)

    os.makedirs(others_dir, exist_ok=True)
    np.save(pixels_path, pixels)
    meta.to_csv(meta_path, index=False)

    print(f"Loaded {len(pixels)} rows from {os.path.basename(csv_path)}")

    return pixels, meta


def to_features(pixels):
    """(n, 2304) uint8 -> float32 in 0-1."""

    return pixels.astype(np.float32) / 255.0


def as_images(pixels):
    """(n, 2304) -> (n, 48, 48), for showing faces."""

    return pixels.reshape(-1, IMG_SIZE, IMG_SIZE)
