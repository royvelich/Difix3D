import unittest

import torch

from src.sparse_attention import (
    build_sparse_attention_context,
    parse_attention_matrix,
    parse_attention_modes,
    sparse_multiview_attention,
)


class FakeAttention(torch.nn.Module):
    def __init__(self, channels=4, heads=2):
        super().__init__()
        self.heads = heads
        self.to_q = torch.nn.Linear(channels, channels, bias=False)
        self.to_k = torch.nn.Linear(channels, channels, bias=False)
        self.to_v = torch.nn.Linear(channels, channels, bias=False)
        self.to_out = torch.nn.ModuleList(
            [torch.nn.Linear(channels, channels, bias=False), torch.nn.Identity()]
        )
        self.group_norm = None
        self.spatial_norm = None
        self.residual_connection = False
        self.rescale_output_factor = 1.0


def dense_reference(attention, hidden_states, context, spatial_shape):
    batch_size, _, channels = hidden_states.shape
    height, width = spatial_shape
    tokens = height * width
    views = hidden_states.reshape(batch_size, context.num_views, tokens, channels)
    validity = context.validity(
        spatial_shape,
        batch_size=batch_size,
        device=hidden_states.device,
    )
    outputs = torch.zeros_like(views)

    for batch_index in range(batch_size):
        for query_view in range(context.num_views):
            query_mode = context.query_modes[query_view]
            if query_mode == "none":
                continue
            query_keep = (
                torch.ones(tokens, dtype=torch.bool)
                if query_mode == "all"
                else validity[batch_index, query_view]
            )
            query_indices = torch.nonzero(query_keep).flatten()
            if not query_indices.numel():
                continue

            source_parts = []
            for source_view in range(context.num_views):
                key_mode = context.key_modes[query_view][source_view]
                if key_mode == "none":
                    continue
                key_keep = (
                    torch.ones(tokens, dtype=torch.bool)
                    if key_mode == "all"
                    else validity[batch_index, source_view]
                )
                source_parts.append(
                    views[batch_index, source_view][key_keep]
                )
            if not source_parts:
                continue

            query_tokens = views[batch_index, query_view][query_indices]
            source_tokens = torch.cat(source_parts)
            query = attention.to_q(query_tokens).view(
                1, -1, attention.heads, channels // attention.heads
            ).transpose(1, 2)
            key = attention.to_k(source_tokens).view(
                1, -1, attention.heads, channels // attention.heads
            ).transpose(1, 2)
            value = attention.to_v(source_tokens).view(
                1, -1, attention.heads, channels // attention.heads
            ).transpose(1, 2)
            attended = torch.nn.functional.scaled_dot_product_attention(
                query, key, value
            )
            attended = attended.transpose(1, 2).reshape(1, -1, channels)
            attended = attention.to_out[0](attended).squeeze(0)
            outputs[batch_index, query_view] = outputs[
                batch_index, query_view
            ].index_copy(0, query_indices, attended)
    return outputs.reshape_as(hidden_states)


class SparseAttentionTest(unittest.TestCase):
    def test_policy_parsing_and_validation(self):
        self.assertEqual(
            parse_attention_modes("all, none,valid_only"),
            ("all", "none", "valid_only"),
        )
        self.assertEqual(
            parse_attention_matrix("all,none;valid_only,all"),
            (("all", "none"), ("valid_only", "all")),
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_attention_modes("all,bad")
        with self.assertRaisesRegex(ValueError, "must contain 3"):
            parse_attention_matrix("all,all;all,all", expected=3)

    def test_validity_pyramid_follows_unet_shapes(self):
        validity = torch.ones(2, 1, 72, 128)
        validity[:, :, :, 64:] = 0
        context = build_sparse_attention_context(
            validity,
            num_views=3,
            wrapped_view_index=2,
            latent_shape=(9, 16),
            num_levels=4,
            valid_fraction_threshold=0.8,
            query_modes=("all", "all", "valid_only"),
            key_modes=(
                ("all", "all", "valid_only"),
                ("all", "all", "valid_only"),
                ("all", "all", "valid_only"),
            ),
        )
        self.assertEqual(
            set(context.token_validity_by_shape),
            {(9, 16), (5, 8), (3, 4), (2, 2)},
        )
        base = context.token_validity_by_shape[(9, 16)]
        self.assertTrue(base[:, :2].all())
        wrapped = base[:, 2].reshape(2, 9, 16)
        self.assertTrue(wrapped[:, :, :8].all())
        self.assertFalse(wrapped[:, :, 8:].any())

    def test_sparse_matches_explicit_gathered_reference(self):
        torch.manual_seed(7)
        attention = FakeAttention()
        validity = torch.tensor(
            [[[[1, 1, 0], [1, 0, 0]]]],
            dtype=torch.float32,
        )
        context = build_sparse_attention_context(
            validity,
            num_views=3,
            wrapped_view_index=2,
            latent_shape=(2, 3),
            num_levels=1,
            valid_fraction_threshold=0.5,
            query_modes=("all", "none", "valid_only"),
            key_modes=(
                ("all", "all", "valid_only"),
                ("none", "none", "none"),
                ("all", "none", "valid_only"),
            ),
        )
        hidden_states = torch.randn(1, 18, 4, requires_grad=True)
        actual = sparse_multiview_attention(
            attention,
            hidden_states,
            context=context,
            spatial_shape=(2, 3),
        )
        expected = dense_reference(
            attention,
            hidden_states,
            context,
            (2, 3),
        )
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(actual[:, 6:12], torch.zeros_like(actual[:, 6:12])))
        invalid_wrapped = actual[:, [14, 16, 17]]
        self.assertTrue(
            torch.equal(
                invalid_wrapped,
                torch.zeros_like(invalid_wrapped),
            )
        )

        actual.sum().backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())

    def test_empty_wrapped_mask_is_finite(self):
        attention = FakeAttention()
        context = build_sparse_attention_context(
            torch.zeros(1, 1, 2, 2),
            num_views=3,
            wrapped_view_index=2,
            latent_shape=(2, 2),
            num_levels=1,
            valid_fraction_threshold=0.8,
            query_modes=("all", "all", "valid_only"),
            key_modes=(
                ("all", "all", "valid_only"),
                ("all", "all", "valid_only"),
                ("all", "all", "valid_only"),
            ),
        )
        output = sparse_multiview_attention(
            attention,
            torch.randn(1, 12, 4),
            context=context,
            spatial_shape=(2, 2),
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[:, 8:], torch.zeros_like(output[:, 8:])))

    def test_autocast_scatter_preserves_input_dtype(self):
        attention = FakeAttention()
        context = build_sparse_attention_context(
            torch.ones(1, 1, 2, 2),
            num_views=3,
            wrapped_view_index=2,
            latent_shape=(2, 2),
            num_levels=1,
            valid_fraction_threshold=0.8,
            query_modes=("all", "all", "valid_only"),
            key_modes=(
                ("all", "all", "valid_only"),
                ("all", "all", "valid_only"),
                ("all", "all", "valid_only"),
            ),
        )
        hidden_states = torch.randn(1, 12, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = sparse_multiview_attention(
                attention,
                hidden_states,
                context=context,
                spatial_shape=(2, 2),
            )
        self.assertEqual(output.dtype, hidden_states.dtype)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
