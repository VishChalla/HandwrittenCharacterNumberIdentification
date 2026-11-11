"""Tests for image preparation. They need only NumPy and Pillow, not TensorFlow."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from digits.preprocess import to_batch, to_mnist_format


def draw_one(size=200, ink=0, paper=255, offset=(0, 0)) -> Image.Image:
    """A vertical stroke: a crude digit 1."""
    image = Image.new("L", (size, size), paper)
    x = size // 2 + offset[0]
    ImageDraw.Draw(image).line([(x, 40 + offset[1]), (x, 160 + offset[1])], fill=ink, width=14)
    return image


def test_output_shape_and_range():
    out = to_mnist_format(draw_one())
    assert out.shape == (28, 28)
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert to_batch(draw_one()).shape == (1, 28, 28, 1)


def test_dark_on_light_is_inverted_to_match_mnist():
    out = to_mnist_format(draw_one(ink=0, paper=255))
    assert out[0, 0] < 0.1  # background is black
    assert out[14, 14] > 0.5  # stroke is bright


def test_light_on_dark_is_left_alone():
    a = to_mnist_format(draw_one(ink=0, paper=255))
    b = to_mnist_format(draw_one(ink=255, paper=0))
    assert np.abs(a - b).max() < 0.05


def test_digit_is_centred_wherever_it_was_drawn():
    centred = to_mnist_format(draw_one())
    shifted = to_mnist_format(draw_one(offset=(45, 20)))
    assert np.abs(centred - shifted).max() < 0.15


def test_digit_fits_the_20_pixel_box():
    out = to_mnist_format(draw_one())
    rows = np.where(out.max(axis=1) > 0.25)[0]
    assert rows.max() - rows.min() + 1 <= 20
    assert rows.min() >= 4 and rows.max() <= 23


def test_blank_image_gives_blank_output():
    assert to_mnist_format(Image.new("L", (100, 100), 255)).sum() == 0


def test_bundled_sample_image_loads():
    sample = Path(__file__).parent / "sample_digit.jpg"
    out = to_mnist_format(Image.open(sample))
    assert out.shape == (28, 28) and out.max() > 0.5


def test_bright_frame_around_a_dark_image_is_ignored():
    """A screenshot of a white-on-black digit with a white margin."""
    plain = draw_one(ink=255, paper=0)
    framed = Image.new("L", (260, 260), 255)
    framed.paste(plain, (30, 30))
    assert np.abs(to_mnist_format(plain) - to_mnist_format(framed)).max() < 0.1
