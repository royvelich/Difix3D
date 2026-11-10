"""Differentiable preprocessing for the render conditioning endpoint."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def smooth_render_endpoint(
    image,
    *,
    mode="none",
    kernel_size=5,
    sigma=1.0,
    strength=1.0,
):
    """Apply a shape-preserving low-pass filter to a BCHW image tensor.

    ``strength`` interpolates between the original and filtered images.
    Reflect padding avoids artificial dark borders for normal images; tiny
    test inputs fall back to replicate padding when reflection is undefined.
    """

    mode = str(mode)
    kernel_size = int(kernel_size)
    sigma = float(sigma)
    strength = float(strength)
    if mode not in ("none", "gaussian", "box"):
        raise ValueError(
            f"render smoothing mode must be none|gaussian|box, got {mode!r}"
        )
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError(
            "render smoothing kernel size must be a positive odd integer"
        )
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("render smoothing sigma must be finite and > 0")
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("render smoothing strength must be in [0, 1]")
    if mode == "none" or kernel_size == 1 or strength == 0.0:
        return image
    if image.ndim != 4:
        raise ValueError(
            "render smoothing expects BCHW input, got "
            f"shape {tuple(image.shape)}"
        )

    radius = kernel_size // 2
    padding_mode = (
        "reflect"
        if image.shape[-2] > radius and image.shape[-1] > radius
        else "replicate"
    )
    padded = F.pad(
        image,
        (radius, radius, radius, radius),
        mode=padding_mode,
    )
    if mode == "box":
        filtered = F.avg_pool2d(
            padded,
            kernel_size=kernel_size,
            stride=1,
        )
    else:
        coordinates = torch.arange(
            kernel_size,
            device=image.device,
            dtype=torch.float32,
        ) - radius
        kernel_1d = torch.exp(
            -0.5 * coordinates.square() / (sigma * sigma)
        )
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d).to(
            dtype=image.dtype
        )
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(
            image.shape[1],
            1,
            kernel_size,
            kernel_size,
        )
        filtered = F.conv2d(
            padded,
            kernel,
            groups=image.shape[1],
        )
    return torch.lerp(image, filtered, strength)
