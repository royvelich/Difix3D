import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_difix import (  # noqa: E402
    _aggregate_metric_rows,
    _log_image,
    _log_metric_scalars,
)
from visualization import build_labeled_grid  # noqa: E402


class FakeClearMLTracker:
    name = "clearml"

    def __init__(self):
        self.images = []
        self.scalars = []

    def log_images(self, values, step):
        self.images.append((values, step))

    def log(self, values, step):
        self.scalars.append((values, step))


class ValidationLoggingTest(unittest.TestCase):
    def test_exact_mean_median_and_clearml_tags(self):
        rows = [
            {
                "metrics": {
                    "ours_ref_psnr": 10.0,
                    "input_psnr": 8.0,
                }
            },
            {
                "metrics": {
                    "ours_ref_psnr": 14.0,
                    "input_psnr": 10.0,
                }
            },
        ]
        average, median, counts = _aggregate_metric_rows(rows)

        self.assertEqual(average["ours_ref_psnr"], 12.0)
        self.assertEqual(median["input_psnr"], 9.0)
        self.assertEqual(counts["ours_ref_psnr"], 2)

        tracker = FakeClearMLTracker()
        _log_metric_scalars(tracker, average, median, step=5000)
        logged = tracker.scalars[0][0]
        self.assertEqual(
            logged["restore-val/metrics/ours_ref_psnr"],
            12.0,
        )
        self.assertEqual(
            logged["restore-val/metrics/input_psnr_median"],
            9.0,
        )

    def test_labeled_panel_upload(self):
        panel = np.zeros((8, 12, 3), dtype=np.uint8)
        grid = build_labeled_grid(
            [{"panels": [panel] * 5, "caption": "pair-0"}],
            ["corrupt", "reference", "clean", "render", "restored ref"],
        )
        tracker = FakeClearMLTracker()
        _log_image(tracker, "restore-val/pair_00", grid, step=5000)

        self.assertEqual(grid.shape, (82, 60, 3))
        self.assertEqual(tracker.images[0][1], 5000)
        np.testing.assert_array_equal(
            tracker.images[0][0]["restore-val/pair_00"],
            grid,
        )


if __name__ == "__main__":
    unittest.main()
