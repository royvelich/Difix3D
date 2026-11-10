import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from render_preprocessing import smooth_render_endpoint  # noqa: E402


class RenderPreprocessingTest(unittest.TestCase):
    def test_gaussian_smoothing_preserves_shape_and_constant_images(self):
        image = torch.full((2, 3, 9, 11), 0.25)

        smoothed = smooth_render_endpoint(
            image,
            mode="gaussian",
            kernel_size=5,
            sigma=2.0,
        )

        self.assertEqual(smoothed.shape, image.shape)
        torch.testing.assert_close(smoothed, image)

    def test_strength_interpolates_between_input_and_filtered_image(self):
        image = torch.zeros(1, 3, 7, 7)
        image[..., 3, 3] = 1.0
        filtered = smooth_render_endpoint(
            image,
            mode="gaussian",
            kernel_size=3,
            sigma=1.0,
            strength=1.0,
        )

        halfway = smooth_render_endpoint(
            image,
            mode="gaussian",
            kernel_size=3,
            sigma=1.0,
            strength=0.5,
        )

        torch.testing.assert_close(
            halfway,
            torch.lerp(image, filtered, 0.5),
        )
        self.assertLess(float(filtered[..., 3, 3].max()), 1.0)
        self.assertGreater(float(filtered[..., 3, 2].min()), 0.0)

    def test_box_smoothing_supports_inputs_smaller_than_the_kernel_radius(self):
        image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)

        smoothed = smooth_render_endpoint(
            image,
            mode="box",
            kernel_size=5,
            sigma=1.0,
        )

        self.assertEqual(smoothed.shape, image.shape)
        self.assertTrue(torch.isfinite(smoothed).all())

    def test_disabled_smoothing_returns_the_original_tensor(self):
        image = torch.randn(1, 3, 4, 4)

        self.assertIs(
            smooth_render_endpoint(image, mode="none"),
            image,
        )
        self.assertIs(
            smooth_render_endpoint(
                image,
                mode="gaussian",
                kernel_size=3,
                sigma=1.0,
                strength=0.0,
            ),
            image,
        )

    def test_invalid_parameters_are_rejected(self):
        image = torch.zeros(1, 3, 4, 4)

        with self.assertRaisesRegex(ValueError, "positive odd"):
            smooth_render_endpoint(
                image,
                mode="gaussian",
                kernel_size=4,
            )
        with self.assertRaisesRegex(ValueError, "finite and > 0"):
            smooth_render_endpoint(
                image,
                mode="gaussian",
                sigma=0.0,
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            smooth_render_endpoint(
                image,
                mode="gaussian",
                strength=1.1,
            )


if __name__ == "__main__":
    unittest.main()
