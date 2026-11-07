"""Sparse multi-view attention helpers for the DiFix SD-Turbo U-Net.

The U-Net remains dense because its convolutions require complete feature
grids. Only ``attn1`` gathers the query/key/value positions allowed by the
configured view policy. Attention outputs are scattered back to the dense grid
before the existing residual, text-attention, feed-forward, and convolutional
paths continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


ATTENTION_MASK_MODES = ("all", "valid_only", "none")


def parse_attention_modes(
    value: str | Sequence[str],
    *,
    expected: int | None = None,
    name: str = "attention modes",
) -> Tuple[str, ...]:
    if isinstance(value, str):
        modes = tuple(item.strip() for item in value.split(","))
    else:
        modes = tuple(str(item).strip() for item in value)
    if not modes or any(not item for item in modes):
        raise ValueError(f"{name} must be a non-empty comma-separated list")
    invalid = sorted(set(modes) - set(ATTENTION_MASK_MODES))
    if invalid:
        raise ValueError(
            f"{name} contains unsupported modes {invalid}; expected "
            f"{ATTENTION_MASK_MODES}"
        )
    if expected is not None and len(modes) != expected:
        raise ValueError(
            f"{name} must contain {expected} entries, got {len(modes)}"
        )
    return modes


def parse_attention_matrix(
    value: str | Sequence[Sequence[str]],
    *,
    expected: int | None = None,
    name: str = "view attention matrix",
) -> Tuple[Tuple[str, ...], ...]:
    if isinstance(value, str):
        raw_rows = [row.strip() for row in value.split(";")]
        if not raw_rows or any(not row for row in raw_rows):
            raise ValueError(
                f"{name} must use semicolon-separated, non-empty rows"
            )
        rows = tuple(
            parse_attention_modes(row, name=f"{name} row {index}")
            for index, row in enumerate(raw_rows)
        )
    else:
        rows = tuple(
            parse_attention_modes(row, name=f"{name} row {index}")
            for index, row in enumerate(value)
        )

    size = len(rows) if expected is None else expected
    if len(rows) != size:
        raise ValueError(f"{name} must contain {size} rows, got {len(rows)}")
    for index, row in enumerate(rows):
        if len(row) != size:
            raise ValueError(
                f"{name} row {index} must contain {size} entries, "
                f"got {len(row)}"
            )
    return rows


@dataclass(frozen=True)
class SparseMultiViewAttentionContext:
    """Per-forward routing and multiresolution validity state.

    ``token_validity_by_shape[(h, w)]`` is Boolean ``[B,V,h*w]``. Views that
    do not have a partial-validity mask are represented as all-valid.
    """

    num_views: int
    query_modes: Tuple[str, ...]
    key_modes: Tuple[Tuple[str, ...], ...]
    token_validity_by_shape: Dict[Tuple[int, int], torch.Tensor]
    wrapped_view_index: int
    valid_fraction_threshold: float

    def __post_init__(self):
        if self.num_views < 1:
            raise ValueError("num_views must be positive")
        parse_attention_modes(
            self.query_modes,
            expected=self.num_views,
            name="query modes",
        )
        parse_attention_matrix(
            self.key_modes,
            expected=self.num_views,
            name="key modes",
        )
        if not 0.0 <= self.valid_fraction_threshold <= 1.0:
            raise ValueError("valid_fraction_threshold must be in [0,1]")
        if not 0 <= self.wrapped_view_index < self.num_views:
            raise ValueError(
                "wrapped_view_index must identify one of the configured views"
            )

    def validity(
        self,
        spatial_shape: Tuple[int, int],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        shape = tuple(int(value) for value in spatial_shape)
        if shape not in self.token_validity_by_shape:
            available = ", ".join(
                str(key) for key in sorted(self.token_validity_by_shape)
            )
            raise ValueError(
                f"No sparse-attention validity mask for U-Net shape {shape}; "
                f"available shapes: {available}"
            )
        validity = self.token_validity_by_shape[shape]
        expected = (batch_size, self.num_views, shape[0] * shape[1])
        if tuple(validity.shape) != expected:
            raise ValueError(
                "Sparse-attention validity shape mismatch: expected "
                f"{expected}, got {tuple(validity.shape)}"
            )
        return validity.to(device=device, dtype=torch.bool)


def _pool_pixels_to_latents(
    pixel_validity: torch.Tensor,
    latent_shape: Tuple[int, int],
) -> torch.Tensor:
    pixel_height, pixel_width = pixel_validity.shape[-2:]
    latent_height, latent_width = latent_shape
    if (
        pixel_height % latent_height == 0
        and pixel_width % latent_width == 0
    ):
        kernel = (
            pixel_height // latent_height,
            pixel_width // latent_width,
        )
        return F.avg_pool2d(
            pixel_validity.float(),
            kernel_size=kernel,
            stride=kernel,
        )
    return F.adaptive_avg_pool2d(
        pixel_validity.float(),
        output_size=latent_shape,
    )


def build_sparse_attention_context(
    pixel_validity: torch.Tensor,
    *,
    num_views: int,
    wrapped_view_index: int,
    latent_shape: Tuple[int, int],
    num_levels: int,
    valid_fraction_threshold: float,
    query_modes: Sequence[str],
    key_modes: Sequence[Sequence[str]],
) -> SparseMultiViewAttentionContext:
    """Build masks aligned to all attention-bearing U-Net resolutions.

    The first mask pools pixels to the actual VAE latent shape. Deeper masks
    follow the U-Net's stride-2, kernel-3, padding-1 downsampling geometry.
    Up-path attention reuses the mask for the matching spatial resolution.
    """

    if pixel_validity.ndim != 4 or pixel_validity.shape[1] != 1:
        raise ValueError(
            "wrapped-reference validity must be [B,1,H,W], got "
            f"{tuple(pixel_validity.shape)}"
        )
    if num_levels < 1:
        raise ValueError("num_levels must be positive")
    if not 0.0 <= valid_fraction_threshold <= 1.0:
        raise ValueError("valid_fraction_threshold must be in [0,1]")
    if wrapped_view_index < 0:
        wrapped_view_index += num_views
    if not 0 <= wrapped_view_index < num_views:
        raise ValueError("wrapped_view_index is outside the view dimension")

    query_modes = parse_attention_modes(
        query_modes,
        expected=num_views,
        name="query modes",
    )
    key_modes = parse_attention_matrix(
        key_modes,
        expected=num_views,
        name="key modes",
    )

    fractions = _pool_pixels_to_latents(pixel_validity, latent_shape)
    masks: Dict[Tuple[int, int], torch.Tensor] = {}
    batch_size = pixel_validity.shape[0]

    for level in range(num_levels):
        height, width = fractions.shape[-2:]
        wrapped_validity = (
            fractions >= float(valid_fraction_threshold)
        ).flatten(1)
        view_validity = torch.ones(
            batch_size,
            num_views,
            height * width,
            dtype=torch.bool,
            device=pixel_validity.device,
        )
        view_validity[:, wrapped_view_index] = wrapped_validity
        masks[(height, width)] = view_validity

        if level + 1 < num_levels:
            fractions = F.avg_pool2d(
                fractions,
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=False,
            )

    return SparseMultiViewAttentionContext(
        num_views=num_views,
        query_modes=query_modes,
        key_modes=key_modes,
        token_validity_by_shape=masks,
        wrapped_view_index=wrapped_view_index,
        valid_fraction_threshold=float(valid_fraction_threshold),
    )


def _selected_indices(mode: str, validity: torch.Tensor) -> torch.Tensor:
    if mode == "all":
        return torch.arange(validity.numel(), device=validity.device)
    if mode == "valid_only":
        return torch.nonzero(validity, as_tuple=False).flatten()
    if mode == "none":
        return torch.empty(0, dtype=torch.long, device=validity.device)
    raise ValueError(f"Unsupported sparse-attention mode {mode!r}")


def sparse_multiview_attention(
    attention,
    hidden_states: torch.Tensor,
    *,
    context: SparseMultiViewAttentionContext,
    spatial_shape: Tuple[int, int],
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """Apply split-query sparse attention and return a dense joint sequence.

    Args:
        attention: Diffusers ``Attention`` module used by ``attn1``.
        hidden_states: normalized joint tokens ``[B,V*N,C]``.
        context: per-forward view routing and validity pyramid.
        spatial_shape: exact ``(h,w)`` captured before Transformer2D flattening.
    """

    if hidden_states.ndim != 3:
        raise ValueError(
            "sparse attention expects [B,V*N,C], got "
            f"{tuple(hidden_states.shape)}"
        )
    batch_size, joint_length, channels = hidden_states.shape
    num_views = context.num_views
    height, width = (int(value) for value in spatial_shape)
    tokens_per_view = height * width
    if joint_length != num_views * tokens_per_view:
        raise ValueError(
            "joint token count does not match views and spatial shape: "
            f"{joint_length} != {num_views}*{height}*{width}"
        )
    if getattr(attention, "spatial_norm", None) is not None:
        raise ValueError(
            "Sparse DiFix attn1 does not support Attention.spatial_norm"
        )

    attention_input = hidden_states
    if getattr(attention, "group_norm", None) is not None:
        attention_input = attention.group_norm(
            attention_input.transpose(1, 2)
        ).transpose(1, 2)

    per_view = attention_input.reshape(
        batch_size,
        num_views,
        tokens_per_view,
        channels,
    )
    validity = context.validity(
        (height, width),
        batch_size=batch_size,
        device=hidden_states.device,
    )
    module_args = () if getattr(attention, "_from_deprecated_attn_block", False) else ()
    # Diffusers 0.25.1 uses PEFT adapters without a positional scale argument.
    # Non-PEFT attention modules accept it only in legacy LoRA layers.
    try:
        from diffusers.utils import USE_PEFT_BACKEND

        module_args = () if USE_PEFT_BACKEND else (lora_scale,)
    except ImportError:
        module_args = ()

    batch_outputs = []
    for batch_index in range(batch_size):
        view_outputs = []
        for query_view in range(num_views):
            query_indices = _selected_indices(
                context.query_modes[query_view],
                validity[batch_index, query_view],
            )
            dense_output = torch.zeros_like(
                per_view[batch_index, query_view]
            )
            if query_indices.numel() == 0:
                view_outputs.append(dense_output)
                continue

            key_tokens = []
            for source_view in range(num_views):
                mode = context.key_modes[query_view][source_view]
                key_indices = _selected_indices(
                    mode,
                    validity[batch_index, source_view],
                )
                if key_indices.numel() == 0:
                    continue
                key_tokens.append(
                    per_view[batch_index, source_view].index_select(
                        0, key_indices
                    )
                )

            if not key_tokens:
                view_outputs.append(dense_output)
                continue

            query_tokens = per_view[
                batch_index, query_view
            ].index_select(0, query_indices)
            source_tokens = torch.cat(key_tokens, dim=0)
            query = attention.to_q(
                query_tokens.unsqueeze(0), *module_args
            )
            key = attention.to_k(
                source_tokens.unsqueeze(0), *module_args
            )
            value = attention.to_v(
                source_tokens.unsqueeze(0), *module_args
            )

            heads = int(attention.heads)
            head_dim = key.shape[-1] // heads
            query = query.view(1, -1, heads, head_dim).transpose(1, 2)
            key = key.view(1, -1, heads, head_dim).transpose(1, 2)
            value = value.view(1, -1, heads, head_dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).reshape(
                1, -1, heads * head_dim
            )
            attended = attended.to(query.dtype)
            attended = attention.to_out[0](attended, *module_args)
            attended = attention.to_out[1](attended)
            if getattr(attention, "residual_connection", False):
                attended = attended + query_tokens.unsqueeze(0)
            attended = attended / float(
                getattr(attention, "rescale_output_factor", 1.0)
            )
            dense_output = dense_output.index_copy(
                0,
                query_indices,
                attended.squeeze(0).to(dense_output.dtype),
            )
            view_outputs.append(dense_output)
        batch_outputs.append(torch.stack(view_outputs, dim=0))

    return torch.stack(batch_outputs, dim=0).reshape(
        batch_size,
        joint_length,
        channels,
    )
