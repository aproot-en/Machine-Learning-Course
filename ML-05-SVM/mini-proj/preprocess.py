import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """Resize one image and convert it to grayscale. None if unusable."""

    if image is None or image.size == 0:
        return None

    # Convert BGR to grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image (INTER_AREA is the right filter for shrinking)
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):
    """(n, h, w) uint8 -> (n, h*w) float32 in 0-1."""

    features = images.reshape(len(images), -1).astype(np.float32)
    # Normalize pixel values from 0-255 to 0-1
    features /= 255.0

    return features


def preprocess_images(images, img_size=100):
    """Raw image list -> feature matrix (for small datasets)."""

    processed = [preprocess_image(img, img_size) for img in images]
    processed = [img for img in processed if img is not None]

    return to_features(np.stack(processed))
