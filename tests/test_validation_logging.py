import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_difix import (  # noqa: E402
    _StridedValidationSampler,
    _aggregate_metric_rows,
    _dataset_kwargs,
    _image_panel,
    _log_image,
    _log_metric_scalars,
    _prepare_model_conditioning,
    _valid_pixel_metrics,
    _wrapped_loss_tensors,
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
    def test_validation_sampler_partitions_without_duplicates(self):
        dataset = list(range(10))
        partitions = [
            list(_StridedValidationSampler(dataset, rank, 3))
            for rank in range(3)
        ]

        self.assertEqual(partitions[0], [0, 3, 6, 9])
        self.assertEqual(partitions[1], [1, 4, 7])
        self.assertEqual(partitions[2], [2, 5, 8])
        self.assertEqual(
            sorted(index for partition in partitions for index in partition),
            list(range(10)),
        )

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

    def test_named_validation_metric_namespace(self):
        tracker = FakeClearMLTracker()
        _log_metric_scalars(
            tracker,
            {"ours_ref_lpips": 0.25},
            {"ours_ref_lpips": 0.2},
            step=250,
            namespace="restore-val/historical100",
        )

        logged = tracker.scalars[0][0]
        self.assertEqual(
            logged["restore-val/historical100/metrics/ours_ref_lpips"],
            0.25,
        )
        self.assertEqual(
            logged[
                "restore-val/historical100/metrics/ours_ref_lpips_median"
            ],
            0.2,
        )

    def test_validation_dataset_can_override_path_remapping(self):
        args = SimpleNamespace(
            height=576,
            width=1024,
            prompt=None,
            default_prompt="remove degradation",
            data_root="/train",
            path_prefix_from="/primary/source",
            path_prefix_to="/primary/target",
            wrapped_reference_fill="black",
            wrapped_reference_builder_repo=None,
            wrapped_reference_depth_cache_dir=None,
            wrapped_reference_splat_radius=1,
            cache_built_wrapped_references=False,
            storage_backend="local",
            s3_cache_dir=None,
            s3_cache_images=False,
            s3_max_pool_connections=32,
            s3_tar_routes_path=None,
        )

        values = _dataset_kwargs(
            args,
            data_root="/historical",
            path_prefix_from="/historical/source",
            path_prefix_to="/historical/target",
        )

        self.assertEqual(values["data_root"], "/historical")
        self.assertEqual(
            values["path_prefix_from"],
            "/historical/source",
        )
        self.assertEqual(
            values["path_prefix_to"],
            "/historical/target",
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

    def test_wrapped_loss_masks_invalid_pixels_without_dilution(self):
        prediction = torch.zeros(1, 3, 3, 2, 2)
        target = torch.zeros_like(prediction)
        prediction[:, 2, :, 0, 0] = 2.0
        prediction[:, 2, :, 1, 1] = 100.0
        validity = torch.zeros(1, 1, 2, 2)
        validity[:, :, 0, 0] = 1.0

        l2, perceptual_prediction, target_flat, weights = (
            _wrapped_loss_tensors(
                prediction,
                target,
                validity,
                "valid_pixels",
            )
        )

        self.assertAlmostEqual(float(l2), 4.0 / 3.0)
        self.assertTrue(
            torch.equal(
                perceptual_prediction[2, :, 1, 1],
                target_flat[2, :, 1, 1],
            )
        )
        torch.testing.assert_close(weights, torch.ones(3))

    def test_oracle_conditioning_is_shared_by_train_and_validation(self):
        conditioning = torch.zeros(1, 3, 3, 2, 4)
        conditioning[:, 1] = 0.25
        conditioning[:, 2, :, :, :2] = 0.75
        conditioning[:, 2, :, :, 2:] = -0.5
        target = torch.ones_like(conditioning)
        target[:, 1] = conditioning[:, 1]
        validity = torch.ones(1, 1, 2, 4)
        args = SimpleNamespace(
            render_endpoint_mode="wrapped_patch_override",
            render_endpoint_override_patch_size=2,
            render_endpoint_override_valid_fraction=1.0,
            wrapped_reference_patch_valid_fraction=1.0,
            render_endpoint_override_policy="oracle_gt",
            render_endpoint_oracle_metric="l1",
            render_endpoint_oracle_window_patches=3,
            render_endpoint_oracle_margin=0.0,
            render_endpoint_oracle_relative_gain=0.0,
            render_endpoint_oracle_lpips_chunk_size=16,
        )

        model_conditioning, info = _prepare_model_conditioning(
            conditioning,
            target,
            validity,
            args,
        )

        self.assertTrue(
            torch.all(model_conditioning[:, 0, :, :, :2] == 0.75)
        )
        self.assertTrue(
            torch.all(model_conditioning[:, 0, :, :, 2:] == 0.0)
        )
        self.assertTrue(torch.all(conditioning[:, 0] == 0.0))
        self.assertAlmostEqual(float(info["selected_fraction"]), 0.5)
        self.assertAlmostEqual(float(info["selected_of_eligible"]), 0.5)

        panel = _image_panel(
            model_conditioning,
            target,
            target,
            0,
            "pair-0",
            validity,
            original_conditioning=conditioning,
            render_override_mask=info["selected_patches"],
        )
        self.assertEqual(panel.ndim, 3)
        self.assertEqual(panel.shape[-1], 3)

    def test_valid_wrapped_metrics_ignore_unsupervised_holes(self):
        prediction = torch.ones(1, 3, 4, 4)
        target = torch.zeros_like(prediction)
        validity = torch.zeros(1, 1, 4, 4)
        validity[..., :2, :2] = 1.0
        prediction[..., 2:, 2:] = 100.0

        with mock.patch(
            "train_difix.batch_metrics",
            return_value={"lpips": 0.25, "psnr": 99.0},
        ) as metric:
            values = _valid_pixel_metrics(
                prediction,
                target,
                validity,
                "restored_wrapped_valid",
            )

        filled_prediction = metric.call_args.args[0]
        self.assertTrue(torch.all(filled_prediction[..., 2:, 2:] == 0.0))
        self.assertAlmostEqual(
            values["restored_wrapped_valid_coverage"],
            0.25,
        )
        self.assertAlmostEqual(
            values["restored_wrapped_valid_lpips"],
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
