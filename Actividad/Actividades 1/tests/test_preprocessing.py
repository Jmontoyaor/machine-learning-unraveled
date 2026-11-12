import unittest

import numpy as np
from PIL import Image, ImageDraw

from mnist_app.preprocessing import preprocess_image


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        black = Image.new("L", (100, 100), 0)
        draw = ImageDraw.Draw(black)
        draw.line((50, 15, 50, 85), fill=255, width=12)
        self.black = black

    def test_equivalent_polarities(self):
        _, a, _ = preprocess_image(self.black)
        _, b, _ = preprocess_image(Image.fromarray(255 - np.asarray(self.black)))
        np.testing.assert_allclose(a, b, atol=1 / 255)
        self.assertEqual(a.shape, (28, 28))
        self.assertEqual(a.dtype, np.float32)
        self.assertGreater(a.max(), 0.9)
        self.assertEqual(a.min(), 0)

    def test_empty_canvas_rejected(self):
        for level in (0, 255):
            with self.assertRaises(ValueError):
                preprocess_image(Image.new("L", (100, 100), level))

    def test_manual_polarity(self):
        _, auto, info = preprocess_image(self.black)
        self.assertFalse(info["inverted"])
        _, manual, info = preprocess_image(self.black, invert=False)
        np.testing.assert_array_equal(auto, manual)

    def test_off_center_digit_is_centered_by_mass(self):
        image = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(image).line((15, 10, 15, 40), fill=255, width=8)
        _, result, _ = preprocess_image(image)
        rows, cols = np.indices(result.shape)
        cy = (rows * result).sum() / result.sum()
        cx = (cols * result).sum() / result.sum()
        self.assertLess(abs(cy - 14), 1.0)
        self.assertLess(abs(cx - 14), 1.0)

    def test_aspect_ratio_is_preserved(self):
        _, result, _ = preprocess_image(self.black)
        cols = np.flatnonzero(result.max(axis=0) > 0.1)
        rows = np.flatnonzero(result.max(axis=1) > 0.1)
        self.assertGreater(len(rows), 2 * len(cols))  # a vertical bar stays tall and thin

    def test_canvas_frame_is_ignored(self):
        image = Image.new("L", (100, 100), 0)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 99, 99), outline=255, width=1)
        draw.line((50, 30, 50, 70), fill=255, width=10)
        _, result, _ = preprocess_image(image)
        self.assertEqual(result[0].max(), 0)
        self.assertEqual(result[:, 0].max(), 0)

    def test_values_in_unit_range(self):
        _, result, info = preprocess_image(self.black)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)
        self.assertIn("final", info["steps"])


if __name__ == "__main__":
    unittest.main()
