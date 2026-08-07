import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """Resize one image to img_size x img_size RGB. None if unusable."""

    if image is None or image.size == 0:
        return None

    # cv2 reads BGR, convert to RGB so images display correctly
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image (INTER_AREA is the right filter for shrinking)
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):
    """(n, h, w, 3) uint8 -> the array the model consumes.

    A CNN keeps the 2D shape, so unlike the SVM version nothing is flattened.

    Data stays uint8 on purpose: the 0-1 scaling is a Rescaling layer inside
    the model instead. float32 RGB would be 4x the memory (720 MB for 6,000
    images vs 180 MB), and keeping it in the model means test_nn.py cannot
    forget to apply it.
    """

    return np.ascontiguousarray(images, dtype=np.uint8)
