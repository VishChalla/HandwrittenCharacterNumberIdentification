"""Turn any photo or scan of a digit into the format the model was trained on.

MNIST digits are white strokes on a black background, scaled to fit a 20x20
box and centred in a 28x28 image. A photo of a digit on paper is the opposite
in almost every way: dark on light, off-centre and any size. Feeding it to the
model unchanged is the most common reason a 99%-accurate MNIST model fails on
real uploads, so this module closes that gap.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 28
BOX = 20  # MNIST digits occupy the central 20x20 pixels


def _strip_bright_frame(gray: np.ndarray) -> np.ndarray:
    top, bottom, left, right = 0, gray.shape[0], 0, gray.shape[1]
    while bottom - top > 8 and gray[top, left:right].mean() > 0.5:
        top += 1
    while bottom - top > 8 and gray[bottom - 1, left:right].mean() > 0.5:
        bottom -= 1
    while right - left > 8 and gray[top:bottom, left].mean() > 0.5:
        left += 1
    while right - left > 8 and gray[top:bottom, right - 1].mean() > 0.5:
        right -= 1
    return gray[top:bottom, left:right]


def to_mnist_format(image: Image.Image) -> np.ndarray:
    """Return a ``(28, 28)`` float array in [0, 1], white digit on black."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    # Paper is light, MNIST backgrounds are dark. The background is whatever
    # most of the image is, so use the median rather than a fixed corner.
    if np.median(gray) > 0.5:
        gray = 1.0 - gray

    # Screenshots often carry a bright frame around a dark image. Strip any
    # border rows and columns that are mostly bright so they are not read as ink.
    gray = _strip_bright_frame(gray)

    # Remove background haze so a grey page does not count as ink.
    gray = np.clip((gray - gray.min()) / max(gray.max() - gray.min(), 1e-6), 0, 1)
    ink = gray > 0.25
    if not ink.any():
        return np.zeros((SIZE, SIZE), dtype=np.float32)

    # Crop to the digit, scale its longer side to 20 pixels, centre it.
    rows, cols = np.where(ink)
    crop = gray[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    scale = BOX / max(crop.shape)
    new_size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    digit = Image.fromarray((crop * 255).astype(np.uint8)).resize(new_size, Image.LANCZOS)
    digit = np.asarray(digit, dtype=np.float32) / 255.0

    canvas = np.zeros((SIZE, SIZE), dtype=np.float32)
    top, left = (SIZE - digit.shape[0]) // 2, (SIZE - digit.shape[1]) // 2
    canvas[top:top + digit.shape[0], left:left + digit.shape[1]] = digit
    return canvas


def to_batch(image: Image.Image) -> np.ndarray:
    """Model input with batch and channel dimensions: ``(1, 28, 28, 1)``."""
    return to_mnist_format(image).reshape(1, SIZE, SIZE, 1)
