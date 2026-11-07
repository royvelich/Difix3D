import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusers.models.attention import BasicTransformerBlock  # noqa: E402
from mv_unet import UNet2DConditionModel, new_forward  # noqa: E402
from sparse_attention import build_sparse_attention_context  # noqa: E402


def tiny_unet():
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        cross_attention_dim=16,
        attention_head_dim=8,
        norm_num_groups=8,
    )


class SparseMvUnetTest(unittest.TestCase):
    def test_patch_is_instance_local_and_dense_two_view_runs(self):
        class_forward = BasicTransformerBlock.forward
        model = tiny_unet()

        self.assertIs(BasicTransformerBlock.forward, class_forward)
        self.assertTrue(model._difix_transformer_blocks)
        for block in model._difix_transformer_blocks:
            self.assertIs(block.forward.__func__, new_forward)

        model.set_multiview_attention_context(None, num_views=2)
        output = model(
            torch.randn(2, 4, 8, 8),
            torch.tensor(199),
            torch.randn(2, 5, 16),
        ).sample
        self.assertEqual(tuple(output.shape), (2, 4, 8, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_sparse_three_view_checkpointed_backward(self):
        model = tiny_unet()
        model.enable_gradient_checkpointing()
        model.train()
        context = build_sparse_attention_context(
            torch.ones(1, 1, 64, 64),
            num_views=3,
            wrapped_view_index=2,
            latent_shape=(8, 8),
            num_levels=1,
            valid_fraction_threshold=0.8,
            query_modes=("all", "all", "valid_only"),
            key_modes=(
                ("all", "all", "valid_only"),
                ("all", "all", "valid_only"),
                ("all", "all", "valid_only"),
            ),
        )
        model.set_multiview_attention_context(context)
        sample = torch.randn(3, 4, 8, 8, requires_grad=True)
        output = model(
            sample,
            torch.tensor(199),
            torch.randn(3, 5, 16),
        ).sample
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (3, 4, 8, 8))
        self.assertIsNotNone(sample.grad)
        self.assertTrue(torch.isfinite(sample.grad).all())


if __name__ == "__main__":
    unittest.main()
