"""Visualisation utilities for sketch-to-image results.

Creates side-by-side comparison grids and experiment summary figures
suitable for the report and README.

All functions accept PIL Images and write to disk via ``pathlib.Path``.
No display side-effects — this module is Colab-safe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core comparison grid
# ---------------------------------------------------------------------------


def save_side_by_side(
    sketch: Image.Image,
    conditioning_map: Image.Image,
    generated: Image.Image,
    output_path: str | Path,
    titles: Sequence[str] = ("Sketch", "Conditioning Map", "Generated"),
    dpi: int = 150,
) -> Path:
    """Save a 3-panel figure: sketch | conditioning map | generated image.

    Args:
        sketch: Original hand-drawn sketch.
        conditioning_map: Preprocessed edge/conditioning map.
        generated: Output image from the diffusion pipeline.
        output_path: Destination file path (PNG).
        titles: Panel titles shown above each image.
        dpi: Figure DPI (150 is a good balance for reports).

    Returns:
        Resolved absolute path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [sketch, conditioning_map, generated]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Saved side-by-side grid: %s", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Multi-result sweep grid
# ---------------------------------------------------------------------------


def save_sweep_grid(
    sketch: Image.Image,
    results: list[dict],
    output_path: str | Path,
    row_key: str = "controlnet_conditioning_scale",
    col_key: str = "controlnet_type",
    dpi: int = 120,
) -> Path:
    """Save a grid comparing generated images across a parameter sweep.

    Each row corresponds to a unique value of ``row_key``;
    each column to a unique value of ``col_key``.

    Args:
        sketch: The common input sketch image.
        results: List of result dicts, each containing keys:
            ``generated`` (PIL Image or Path), ``config`` (dict).
        output_path: Destination file path (PNG).
        row_key: Config key that varies across rows.
        col_key: Config key that varies across columns.
        dpi: Figure DPI.

    Returns:
        Resolved absolute path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect unique values (preserve insertion order via dict)
    row_vals = list(dict.fromkeys(r["config"][row_key] for r in results))
    col_vals = list(dict.fromkeys(r["config"][col_key] for r in results))
    n_rows, n_cols = len(row_vals), len(col_vals)

    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(4 * (n_cols + 1), 4 * n_rows))
    # Normalise to 2D array
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for ri, row_val in enumerate(row_vals):
        # First column: original sketch
        axes[ri, 0].imshow(sketch)
        axes[ri, 0].set_title("Sketch" if ri == 0 else "", fontsize=9)
        axes[ri, 0].set_ylabel(f"{row_key}={row_val}", fontsize=8)
        axes[ri, 0].axis("off")

        for ci, col_val in enumerate(col_vals):
            ax = axes[ri, ci + 1]
            match = next(
                (r for r in results if r["config"][row_key] == row_val and r["config"][col_key] == col_val),
                None,
            )
            if match is not None:
                img = match["generated"]
                if isinstance(img, (str, Path)):
                    img = Image.open(img)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_facecolor("#f0f0f0")
            if ri == 0:
                ax.set_title(f"{col_key}={col_val}", fontsize=9)
            ax.axis("off")

    fig.suptitle(f"Sweep: {row_key} × {col_key}", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved sweep grid: %s", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Single annotated image
# ---------------------------------------------------------------------------


def annotate_image(
    image: Image.Image,
    text: str,
    position: tuple[int, int] = (10, 10),
    font_size: int = 16,
    color: str = "white",
    bg_color: str | None = "black",
) -> Image.Image:
    """Overlay a text annotation on a PIL image.

    Args:
        image: Base PIL image (modified in-place on a copy).
        text: Text to overlay.
        position: ``(x, y)`` pixel coordinates for the text.
        font_size: Font size in pixels.
        color: Text colour (CSS colour name or hex).
        bg_color: Background fill colour behind text, or ``None`` for none.

    Returns:
        New PIL image with the annotation applied.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    if bg_color is not None:
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle(bbox, fill=bg_color)

    draw.text(position, text, fill=color, font=font)
    return out
