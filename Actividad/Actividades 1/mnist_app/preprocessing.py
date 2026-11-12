"""Convert drawings and uploaded images into the canonical MNIST representation.

Canonical representation (the one MNIST really uses):
    background = 0.0 (black), stroke = ~1.0 (white), anti-aliased edges in between.
"""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

EMPTY_MESSAGE = "No useful stroke found. Draw or upload a digit with visible contrast."
MIN_CONTRAST = 20      # uint8 levels between darkest and brightest pixel
INK_THRESHOLD = 24     # uint8 level above which a pixel counts as stroke
MIN_INK_PIXELS = 3
BORDER_MARGIN = 2      # pixels ignored at the canvas frame


def _to_grayscale(source):
    if isinstance(source, Image.Image):
        image = source.copy()
    elif isinstance(source, (bytes, bytearray)):
        image = Image.open(BytesIO(source))
    else:
        image = Image.fromarray(np.asarray(source))
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba)
    return image.convert("L")


def _shift_to_center_of_mass(image28):
    """Translate the digit so its intensity center of mass sits at (14, 14), as in MNIST."""
    total = image28.sum()
    if total <= 0:
        return image28
    rows, cols = np.indices(image28.shape)
    cy = float((rows * image28).sum() / total)
    cx = float((cols * image28).sum() / total)
    dy, dx = int(round(14 - cy)), int(round(14 - cx))
    shifted = np.zeros_like(image28)
    src = image28[max(0, -dy):28 - max(0, dy), max(0, -dx):28 - max(0, dx)]
    shifted[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return shifted


def preprocess_image(source, invert=None):
    """Return (original, canonical 28x28 float32 image, info).

    info contains the polarity decision and the intermediate steps so the
    interface can show how the input was transformed. Raises ValueError for
    empty inputs instead of producing a meaningless prediction.
    """
    original = _to_grayscale(source)
    gray = np.asarray(original, dtype=np.uint8)
    if int(gray.max()) - int(gray.min()) < MIN_CONTRAST:
        raise ValueError(EMPTY_MESSAGE)

    # Polarity: the border is mostly background. A bright border means a
    # light background (black-on-white), which must be inverted.
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    detected_invert = float(np.median(border)) > 127
    should_invert = detected_invert if invert is None else bool(invert)
    ink = 255 - gray if should_invert else gray

    # Ignore the canvas frame so it never becomes part of the digit.
    mask = ink > INK_THRESHOLD
    m = BORDER_MARGIN
    if min(mask.shape) > 4 * m:
        mask[:m, :] = mask[-m:, :] = False
        mask[:, :m] = mask[:, -m:] = False
    if int(mask.sum()) < MIN_INK_PIXELS:
        raise ValueError(EMPTY_MESSAGE)

    ys, xs = np.where(mask)
    cropped = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Fit inside 20x20 keeping the aspect ratio (no stretching), like MNIST.
    crop = Image.fromarray(cropped)
    scale = 20.0 / max(crop.width, crop.height)
    new_size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    resized = crop.resize(new_size, Image.Resampling.LANCZOS)
    boxed = Image.new("L", (28, 28), 0)
    boxed.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))

    boxed_float = np.asarray(boxed, dtype=np.float32) / 255.0   # normalize exactly once
    result = _shift_to_center_of_mass(boxed_float).clip(0.0, 1.0).astype(np.float32)

    info = {
        "inverted": should_invert,
        "detected_invert": detected_invert,
        "steps": {
            "grayscale": gray,
            "polarity": ink,
            "cropped": cropped,
            "boxed": boxed_float,
            "final": result,
        },
    }
    return original, result, info
