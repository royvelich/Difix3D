import sys
import unittest
import importlib.util
from pathlib import Path
from unittest import mock

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import wrapped_patch_oracle as oracle  # noqa: E402


class WrappedPatchOracleTest(unittest.TestCase):
    def test_selected_patch_copies_only_valid_pixels(self):
        render = torch.full((1, 3, 4, 8), 2.0)
        wrapped = torch.ones_like(render)
        validity = torch.zeros(1, 1, 4, 8)
        validity[..., :4, :4] = 1.0
        validity[..., 3, 1:4] = 0.0  # 13/16 valid: eligible at 0.8.
        validity[..., :3, 4:] = 1.0  # 12/16 valid: ineligible at 0.8.

        combined, selected, eligible = (
            oracle.override_render_with_wrapped_patches(
                render,
                wrapped,
                validity,
                patch_size=4,
                min_valid_fraction=0.8,
                selection_policy="all_valid",
            )
        )

        copied = validity[..., :4, :4].bool().expand(-1, 3, -1, -1)
        self.assertTrue(torch.all(combined[..., :4, :4][copied] == 1.0))
        self.assertTrue(torch.all(combined[..., :4, :4][~copied] == 2.0))
        torch.testing.assert_close(combined[..., :4, 4:], render[..., :4, 4:])
        self.assertTrue(selected[..., :4].all())
        self.assertFalse(selected[..., 4:].any())
        torch.testing.assert_close(selected, eligible)

    def test_l1_oracle_selects_only_gt_beneficial_patch(self):
        render = torch.zeros(1, 3, 4, 8)
        clean = torch.ones_like(render)
        wrapped = torch.empty_like(render)
        wrapped[..., :4] = 0.75
        wrapped[..., 4:] = -0.5
        validity = torch.ones(1, 1, 4, 8)
        validity[..., 3, 3] = 0.0

        combined, selected, eligible = (
            oracle.override_render_with_wrapped_patches(
                render,
                wrapped,
                validity,
                patch_size=4,
                min_valid_fraction=0.8,
                selection_policy="oracle_gt",
                clean=clean,
                oracle_metric="l1",
            )
        )

        self.assertTrue(eligible.all())
        self.assertTrue(selected[..., :4].all())
        self.assertFalse(selected[..., 4:].any())
        copied = validity[..., :4].bool().expand(-1, 3, -1, -1)
        self.assertTrue(torch.all(combined[..., :4][copied] == 0.75))
        self.assertTrue(torch.all(combined[..., :4][~copied] == 0.0))
        torch.testing.assert_close(combined[..., 4:], render[..., 4:])

    def test_relative_gain_is_strict(self):
        render = torch.zeros(1, 3, 2, 2)
        clean = torch.ones_like(render)
        wrapped = torch.full_like(render, 0.04)
        validity = torch.ones(1, 1, 2, 2)

        _combined, selected, _eligible = (
            oracle.override_render_with_wrapped_patches(
                render,
                wrapped,
                validity,
                patch_size=2,
                min_valid_fraction=1.0,
                selection_policy="oracle_gt",
                clean=clean,
                oracle_metric="l1",
                oracle_relative_gain=0.05,
            )
        )

        self.assertFalse(selected.any())

    def test_lpips_uses_configured_context_and_gain(self):
        render = torch.zeros(1, 3, 2, 4)
        clean = torch.ones_like(render)
        wrapped = torch.full_like(render, 0.5)
        validity = torch.ones(1, 1, 2, 4)
        wrapped_scores = torch.tensor([[[[0.10, 0.20]]]])
        render_scores = torch.tensor([[[[0.20, 0.20]]]])

        with mock.patch.object(
            oracle,
            "_lpips_patch_scores",
            return_value=(wrapped_scores, render_scores),
        ) as score:
            combined, selected, _eligible = (
                oracle.override_render_with_wrapped_patches(
                    render,
                    wrapped,
                    validity,
                    patch_size=2,
                    min_valid_fraction=1.0,
                    selection_policy="oracle_gt",
                    clean=clean,
                    oracle_metric="lpips",
                    oracle_window_patches=5,
                    oracle_relative_gain=0.05,
                    oracle_lpips_chunk_size=17,
                )
            )

        self.assertEqual(score.call_args.kwargs["window_patches"], 5)
        self.assertEqual(score.call_args.kwargs["chunk_size"], 17)
        self.assertTrue(selected[..., :2].all())
        self.assertFalse(selected[..., 2:].any())
        self.assertTrue(torch.all(combined[..., :2] == 0.5))
        self.assertTrue(torch.all(combined[..., 2:] == 0.0))

    @unittest.skipUnless(
        importlib.util.find_spec("pyiqa") is not None,
        "PyIQA is not installed",
    )
    def test_lpips_internal_score_matches_public_pyiqa_metric(self):
        from metrics import _get_iqa_metric

        torch.manual_seed(11)
        render = torch.rand(1, 3, 32, 32) * 2.0 - 1.0
        wrapped = torch.rand_like(render) * 2.0 - 1.0
        clean = torch.rand_like(render) * 2.0 - 1.0
        validity = torch.ones(1, 1, 32, 32)
        eligible = torch.ones(1, 1, 1, 1, dtype=torch.bool)

        wrapped_score, render_score = oracle._lpips_patch_scores(
            render,
            wrapped,
            validity,
            clean,
            eligible,
            patch_size=32,
            window_patches=1,
            chunk_size=2,
        )
        metric = _get_iqa_metric("lpips-vgg", torch.device("cpu"))
        with torch.inference_mode():
            expected_wrapped = metric(
                (wrapped + 1.0) / 2.0,
                (clean + 1.0) / 2.0,
            )
            expected_render = metric(
                (render + 1.0) / 2.0,
                (clean + 1.0) / 2.0,
            )

        torch.testing.assert_close(
            wrapped_score.reshape(()),
            expected_wrapped.reshape(()),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            render_score.reshape(()),
            expected_render.reshape(()),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_invalid_nonfinite_wrapped_pixel_is_transparent(self):
        render = torch.zeros(1, 3, 2, 2)
        clean = torch.ones_like(render)
        wrapped = torch.full_like(render, 0.75)
        wrapped[..., 1, 1] = torch.nan
        validity = torch.ones(1, 1, 2, 2)
        validity[..., 1, 1] = 0.0

        combined, selected, _eligible = (
            oracle.override_render_with_wrapped_patches(
                render,
                wrapped,
                validity,
                patch_size=2,
                min_valid_fraction=0.75,
                selection_policy="oracle_gt",
                clean=clean,
                oracle_metric="l1",
            )
        )

        self.assertTrue(selected.all())
        self.assertTrue(torch.isfinite(combined).all())
        torch.testing.assert_close(combined[..., 1, 1], render[..., 1, 1])

    def test_oracle_requires_clean_gt(self):
        tensor = torch.zeros(1, 3, 2, 2)
        validity = torch.ones(1, 1, 2, 2)
        with self.assertRaisesRegex(ValueError, "clean GT"):
            oracle.override_render_with_wrapped_patches(
                tensor,
                tensor,
                validity,
                patch_size=2,
                min_valid_fraction=1.0,
                selection_policy="oracle_gt",
            )


if __name__ == "__main__":
    unittest.main()
