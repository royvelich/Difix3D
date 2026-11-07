"""Ground-truth oracle for render/wrapped-reference patch composition."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _normalize_lpips_feature(feature):
    return feature / (
        feature.square().sum(dim=1, keepdim=True).sqrt() + 1e-10
    )


def _context_grid(image, *, patch_size, window_patches):
    radius = window_patches // 2
    padding = radius * patch_size
    context_size = window_patches * patch_size
    if padding:
        image = F.pad(
            image.unsqueeze(0),
            (padding, padding, padding, padding),
            mode="reflect",
        )[0]
    return (
        image.unfold(1, context_size, patch_size)
        .unfold(2, context_size, patch_size)
        .permute(1, 2, 0, 3, 4)
    )


def _lpips_patch_scores(
    render,
    wrapped_reference,
    validity,
    clean,
    eligible_patch_grid,
    *,
    patch_size,
    window_patches,
    chunk_size,
):
    """Return exact PyIQA LPIPS-VGG scores for eligible local contexts."""

    from metrics import _get_iqa_metric

    metric = _get_iqa_metric("lpips-vgg", render.device)
    network = metric.net
    wrapped_scores = torch.full(
        eligible_patch_grid.shape,
        float("inf"),
        device=render.device,
        dtype=torch.float32,
    )
    render_scores = torch.full_like(wrapped_scores, float("inf"))
    effective_wrapped = torch.where(
        validity.detach().to(render.device).gt(0),
        wrapped_reference.detach().to(render.device),
        render.detach(),
    )

    with torch.inference_mode():
        for batch_index in range(render.shape[0]):
            patch_y, patch_x = eligible_patch_grid[
                batch_index, 0
            ].nonzero(as_tuple=True)
            if patch_y.numel() == 0:
                continue

            wrapped_windows = _context_grid(
                effective_wrapped[batch_index].float(),
                patch_size=patch_size,
                window_patches=window_patches,
            )[patch_y, patch_x]
            render_windows = _context_grid(
                render[batch_index].detach().float(),
                patch_size=patch_size,
                window_patches=window_patches,
            )[patch_y, patch_x]
            clean_windows = _context_grid(
                clean[batch_index].detach().to(render.device).float(),
                patch_size=patch_size,
                window_patches=window_patches,
            )[patch_y, patch_x]

            for start in range(0, patch_y.numel(), chunk_size):
                stop = min(start + chunk_size, patch_y.numel())
                count = stop - start
                images = torch.cat(
                    [
                        wrapped_windows[start:stop],
                        render_windows[start:stop],
                        clean_windows[start:stop],
                    ],
                    dim=0,
                )
                features = network.net(network.scaling_layer(images))
                wrapped_chunk = images.new_zeros((count,))
                render_chunk = images.new_zeros((count,))
                for layer_index, feature in enumerate(features):
                    feature = _normalize_lpips_feature(feature)
                    clean_feature = feature[2 * count :]
                    wrapped_difference = (
                        feature[:count] - clean_feature
                    ).square()
                    render_difference = (
                        feature[count : 2 * count] - clean_feature
                    ).square()
                    wrapped_chunk = wrapped_chunk + network.lins[
                        layer_index
                    ](wrapped_difference).mean(dim=(1, 2, 3))
                    render_chunk = render_chunk + network.lins[
                        layer_index
                    ](render_difference).mean(dim=(1, 2, 3))

                wrapped_scores[
                    batch_index,
                    0,
                    patch_y[start:stop],
                    patch_x[start:stop],
                ] = wrapped_chunk.float()
                render_scores[
                    batch_index,
                    0,
                    patch_y[start:stop],
                    patch_x[start:stop],
                ] = render_chunk.float()

    return wrapped_scores, render_scores


def override_render_with_wrapped_patches(
    render,
    wrapped_reference,
    validity,
    *,
    patch_size,
    min_valid_fraction,
    selection_policy="all_valid",
    clean=None,
    oracle_metric="l1",
    oracle_window_patches=3,
    oracle_margin=0.0,
    oracle_relative_gain=0.0,
    oracle_lpips_chunk_size=512,
):
    """Replace selected render pixels with valid wrapped-reference pixels.

    Patch eligibility is determined by the fraction of geometrically valid
    pixels. Invalid wrapped pixels remain transparent even inside a selected
    patch. ``oracle_gt`` uses clean GT only to make a detached hard decision;
    the clean image is never copied into the model input.
    """

    if render.shape != wrapped_reference.shape:
        raise ValueError(
            "render/wrapped-reference shape mismatch: "
            f"{tuple(render.shape)} vs {tuple(wrapped_reference.shape)}"
        )
    expected_validity = (render.shape[0], 1, *render.shape[-2:])
    if tuple(validity.shape) != expected_validity:
        raise ValueError(
            "wrapped-reference validity must be "
            f"{expected_validity}, got {tuple(validity.shape)}"
        )

    patch_size = int(patch_size)
    height, width = render.shape[-2:]
    if patch_size < 1 or height % patch_size or width % patch_size:
        raise ValueError(
            "oracle patch size must divide the image dimensions exactly: "
            f"patch_size={patch_size}, image={(height, width)}"
        )
    min_valid_fraction = float(min_valid_fraction)
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in [0, 1]")

    valid_bool = validity.detach().to(render.device).gt(0)
    valid_fraction = F.avg_pool2d(
        valid_bool.float(),
        kernel_size=patch_size,
        stride=patch_size,
    )
    eligible_patch_grid = valid_fraction >= min_valid_fraction
    selected_patch_grid = eligible_patch_grid

    selection_policy = str(selection_policy)
    if selection_policy not in ("all_valid", "oracle_gt"):
        raise ValueError(
            "wrapped-patch selection policy must be all_valid|oracle_gt"
        )

    if selection_policy == "oracle_gt":
        if clean is None:
            raise ValueError(
                "oracle_gt wrapped-patch replacement requires clean GT"
            )
        if clean.shape != render.shape:
            raise ValueError(
                "render/clean-GT shape mismatch: "
                f"{tuple(render.shape)} vs {tuple(clean.shape)}"
            )

        oracle_metric = str(oracle_metric)
        if oracle_metric not in ("l1", "mse", "lpips", "lpips_3x3"):
            raise ValueError(
                "oracle_metric must be l1|mse|lpips|lpips_3x3"
            )
        oracle_window_patches = int(oracle_window_patches)
        if oracle_metric == "lpips_3x3":
            oracle_metric = "lpips"
            oracle_window_patches = 3
        if (
            oracle_window_patches < 1
            or oracle_window_patches % 2 != 1
        ):
            raise ValueError(
                "oracle_window_patches must be a positive odd integer"
            )

        oracle_margin = float(oracle_margin)
        if not math.isfinite(oracle_margin) or oracle_margin < 0.0:
            raise ValueError("oracle_margin must be finite and >= 0")
        oracle_relative_gain = float(oracle_relative_gain)
        if not (
            math.isfinite(oracle_relative_gain)
            and 0.0 <= oracle_relative_gain < 1.0
        ):
            raise ValueError(
                "oracle_relative_gain must be finite and in [0, 1)"
            )
        oracle_lpips_chunk_size = int(oracle_lpips_chunk_size)
        if oracle_lpips_chunk_size < 1:
            raise ValueError("oracle_lpips_chunk_size must be >= 1")

        render_f = render.detach().float()
        wrapped_f = wrapped_reference.detach().to(render.device).float()
        clean_f = clean.detach().to(render.device).float()
        if oracle_metric == "lpips":
            wrapped_patch_error, render_patch_error = _lpips_patch_scores(
                render_f,
                wrapped_f,
                valid_bool.float(),
                clean_f,
                eligible_patch_grid,
                patch_size=patch_size,
                window_patches=oracle_window_patches,
                chunk_size=oracle_lpips_chunk_size,
            )
        else:
            render_error = render_f - clean_f
            wrapped_error = wrapped_f - clean_f
            if oracle_metric == "l1":
                render_error = render_error.abs()
                wrapped_error = wrapped_error.abs()
            else:
                render_error = render_error.square()
                wrapped_error = wrapped_error.square()
            render_error = render_error.mean(dim=1, keepdim=True)
            wrapped_error = wrapped_error.mean(dim=1, keepdim=True)
            render_error = torch.where(
                valid_bool,
                render_error,
                torch.zeros_like(render_error),
            )
            wrapped_error = torch.where(
                valid_bool,
                wrapped_error,
                torch.zeros_like(wrapped_error),
            )
            denominator = valid_fraction.clamp_min(
                torch.finfo(torch.float32).eps
            )
            render_patch_error = F.avg_pool2d(
                render_error,
                kernel_size=patch_size,
                stride=patch_size,
            ) / denominator
            wrapped_patch_error = F.avg_pool2d(
                wrapped_error,
                kernel_size=patch_size,
                stride=patch_size,
            ) / denominator

        oracle_selected = (
            wrapped_patch_error + oracle_margin
            < render_patch_error * (1.0 - oracle_relative_gain)
        )
        selected_patch_grid = eligible_patch_grid & oracle_selected

    eligible_patches = F.interpolate(
        eligible_patch_grid.float(),
        size=(height, width),
        mode="nearest",
    ).bool()
    selected_patches = F.interpolate(
        selected_patch_grid.float(),
        size=(height, width),
        mode="nearest",
    ).bool()
    copy_mask = selected_patches & valid_bool
    combined = torch.where(
        copy_mask,
        wrapped_reference.to(device=render.device, dtype=render.dtype),
        render,
    )
    return combined, selected_patches, eligible_patches
