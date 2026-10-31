"""PixelDiT-compatible full-reference validation metrics.

The public metric names, PyIQA model identifiers, input range, and aggregation
protocol intentionally match the restoration validation used by PixelDiT.
"""

from __future__ import annotations

import torch


IQA_METRICS = (
    ("psnr", "psnr"),
    ("ssim", "ssimc"),
    ("lpips", "lpips-vgg"),
    ("topiq", "topiq_fr"),
    ("dists", "dists"),
    ("stlpips", "stlpips-vgg"),
)
IQA_METRIC_KEYS = tuple(name for name, _ in IQA_METRICS)
IQA_HIGHER_IS_BETTER = {
    "psnr": True,
    "ssim": True,
    "lpips": False,
    "topiq": True,
    "dists": False,
    "stlpips": False,
}

_IQA_CACHE = {}


def _get_iqa_metric(backend_name: str, device):
    key = (backend_name, str(device))
    if key not in _IQA_CACHE:
        try:
            import pyiqa
        except ImportError as exc:
            raise RuntimeError(
                "Validation requires the pinned pyiqa==0.1.16 package."
            ) from exc

        metric = pyiqa.create_metric(
            backend_name,
            as_loss=False,
            device=device,
        ).eval()
        metric.requires_grad_(False)
        _IQA_CACHE[key] = metric
    return _IQA_CACHE[key]


def iqa_backend_metadata() -> dict:
    import pyiqa

    return {
        "package": "pyiqa",
        "version": pyiqa.__version__,
        "models": dict(IQA_METRICS),
        "higher_is_better": dict(IQA_HIGHER_IS_BETTER),
        "input_range": "[0,1] RGB",
    }


def batch_metrics(restored: torch.Tensor, clean: torch.Tensor) -> dict:
    """Score RGB ``[B,3,H,W]`` tensors supplied in ``[-1,1]``."""
    restored = (restored.float().clamp(-1, 1) + 1.0) / 2.0
    clean = (clean.float().clamp(-1, 1) + 1.0) / 2.0

    scores = {}
    with torch.inference_mode():
        for public_name, backend_name in IQA_METRICS:
            value = _get_iqa_metric(backend_name, restored.device)(
                restored,
                clean,
            )
            if isinstance(value, tuple):
                value = value[0]
            scores[public_name] = float(value.float().mean().item())
    return scores
