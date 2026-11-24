"""Remove duplicate images and filter empty or overfilled samples."""

import hashlib

import numpy as np


def ink_ratio(images: np.ndarray) -> np.ndarray:
    """Fraction of non-zero pixels per image, shape (N, H, W) -> (N,)."""
    n_pixels = images.shape[1] * images.shape[2]
    return (images > 0).sum(axis=(1, 2)) / n_pixels


def duplicate_mask(images: np.ndarray) -> np.ndarray:
    """True where an image is a duplicate of an earlier one (first occurrence kept)."""
    seen = set()
    is_dup = np.zeros(images.shape[0], dtype=bool)
    for i, image in enumerate(images):
        image_hash = hashlib.blake2b(image.tobytes(), digest_size=16).digest()
        if image_hash in seen:
            is_dup[i] = True
        else:
            seen.add(image_hash)
    return is_dup


def clean_dataset(
    X: np.ndarray, y: np.ndarray, min_ink: float = 0.01, max_ink: float = 0.80
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Remove duplicate and blank/near-full images.

    Returns (X_clean, y_clean, kept_index, report) where `kept_index` maps
    each row of the cleaned arrays back to its row in the original arrays.
    """
    n_total = X.shape[0]

    dup = duplicate_mask(X)
    ratio = ink_ratio(X)
    bad_ink = (ratio < min_ink) | (ratio > max_ink)

    drop = dup | bad_ink
    keep = ~drop
    kept_index = np.nonzero(keep)[0]

    report = {
        "n_total": int(n_total),
        "n_duplicates_removed": int(dup.sum()),
        "n_bad_ink_removed": int((bad_ink & ~dup).sum()),
        "n_removed_total": int(drop.sum()),
        "n_kept": int(keep.sum()),
    }
    return X[keep], y[keep], kept_index, report
