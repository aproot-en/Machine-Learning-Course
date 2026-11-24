"""Check CNN15 loading, preprocessing, predictions, and confidence rejection."""

import numpy as np
import pytest

from exam_cnn.models.inference import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_CHECKPOINT_PATH,
    LETTERS,
    UNKNOWN_LABEL,
    LetterInferenceWrapper,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_CHECKPOINT_PATH.exists(),
    reason="no trained checkpoint at exam_cnn/models_export/cnn15_best.pt -- run PHASE 4 training first",
)


@pytest.fixture(scope="module")
def wrapper():
    return LetterInferenceWrapper()


@pytest.fixture(scope="module")
def test_data():
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "exam_cnn/datasets/processed/az_dataset.npz"
    )
    d = np.load(path)
    return d["X_test"], d["y_test"]


def test_known_letter_is_predicted_correctly(wrapper, test_data):
    """Check the model on clearly classified examples from the held-out test split."""
    X_test, y_test = test_data
    target_label = "A"
    target_idx = LETTERS.index(target_label)
    candidates = np.nonzero(y_test == target_idx)[0]
    assert len(candidates) > 0

    results = wrapper.predict_with_confidence([X_test[i] for i in candidates[:20]])
    predicted_labels = [label for label, _, _ in results]
    ## Not every single sample need be correct, but the model should get most
    ## of 20 clean, already-centered training-distribution "A"s right.
    assert predicted_labels.count(target_label) >= 15


def test_blank_image_is_rejected_as_unknown(wrapper):
    """A blank (all-zero) image must never be confidently guessed -- it
    should come back UNKNOWN, not a random letter."""
    blank = np.zeros((28, 28), dtype=np.uint8)
    result = wrapper.classify([blank])[0]
    ## character_normalize returns an all-zero canvas for a blank input, so
    ## softmax is near-uniform -> confidence should fall below the threshold.
    assert result["confidence"] < CONFIDENCE_THRESHOLD
    assert result["label"] == UNKNOWN_LABEL


def test_preprocess_handles_odd_sized_images(wrapper):
    """preprocess() must not crash on images that aren't already 28x28 --
    real ROI crops from scanned sheets won't be a fixed size."""
    for shape in [(10, 10), (50, 80), (28, 28), (1, 1)]:
        img = np.zeros(shape, dtype=np.uint8)
        img[shape[0] // 2, shape[1] // 2] = (
            255  ## single ink pixel, avoids the all-blank path
        )
        out = wrapper.preprocess(img)
        assert out.shape == (28, 28)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0
