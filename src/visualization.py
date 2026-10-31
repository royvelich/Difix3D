"""Labeled image panels shared by training and validation logging."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def to_uint8_panel(tensor) -> np.ndarray:
    image = (
        ((tensor + 1.0) / 2.0)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    return (image * 255).astype(np.uint8)


def _load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _text_band(width, height, text_items, font, background, foreground):
    band = Image.new("RGB", (int(width), int(height)), background)
    drawing = ImageDraw.Draw(band)
    for text, x, centered in text_items:
        bounds = drawing.textbbox((0, 0), text, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        px = int(x - text_width / 2) if centered else int(x)
        py = int((height - text_height) / 2 - bounds[1])
        drawing.text((px, py), text, fill=foreground, font=font)
    return np.asarray(band)


def build_labeled_grid(
    rows: Sequence[dict],
    column_titles: Sequence[str],
    title_height: int = 44,
    caption_height: int = 30,
) -> np.ndarray:
    if not rows:
        raise ValueError("At least one row is required")
    if len(rows[0]["panels"]) != len(column_titles):
        raise ValueError("Panel count must match column title count")

    panel_height, panel_width = rows[0]["panels"][0].shape[:2]
    total_width = panel_width * len(column_titles)
    title_font = _load_font(_FONT_BOLD, max(18, title_height * 3 // 5))
    caption_font = _load_font(_FONT_MONO, max(11, caption_height * 3 // 5))
    parts = [
        _text_band(
            total_width,
            title_height,
            [
                (title, panel_width * (index + 0.5), True)
                for index, title in enumerate(column_titles)
            ],
            title_font,
            (0, 0, 0),
            (240, 240, 240),
        )
    ]
    for row in rows:
        parts.append(
            _text_band(
                total_width,
                caption_height,
                [(row["caption"], 8, False)],
                caption_font,
                (24, 24, 24),
                (240, 240, 240),
            )
        )
        parts.append(np.concatenate(row["panels"], axis=1))
    return np.concatenate(parts, axis=0)


def save_grid(grid: np.ndarray, output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(grid).save(output_path)
    return output_path
