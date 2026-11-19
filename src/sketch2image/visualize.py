"""Publication-quality visualization utilities for sketch-to-image experiments.

Features:
- :func:`create_experiment_grid`: Generates publication-ready figures displaying the input
  sketch on the far left, followed by generated variants across labelled parameter columns.
- :func:`create_hero_figure`: Constructs a high-impact composite figure comparing each input sketch
  with its best generated result for inclusion in the thesis/report and README.
- :func:`save_side_by_side`: Creates 3-panel figures (Sketch | Condition | Output).
- :func:`save_sweep_grid`: 2D multi-row parameter grid generator.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from PIL import Image

logger = logging.getLogger(__name__)


def _subplots(nrows: int, ncols: int, **kwargs: Any) -> tuple[Figure, Any]:
    """Create a figure + axes without pyplot.

    These figures are only saved to disk, so they need no GUI backend. pyplot
    would pick one (TkAgg on Windows), which fails on a broken Tcl/Tk install
    and crashes when called from worker threads (e.g. the Gradio app).
    """
    fig = Figure(figsize=kwargs.pop("figsize", None))
    return fig, fig.subplots(nrows, ncols, **kwargs)


# ---------------------------------------------------------------------------
# 1. Publication-Quality Experiment Grid (Sketch on Left + Parameter Columns)
# ---------------------------------------------------------------------------


def create_experiment_grid(
    sketch: str | Path | Image.Image,
    generated_images: Sequence[str | Path | Image.Image],
    column_labels: Sequence[str],
    output_path: str | Path,
    title: str = "",
    subtitle: str | None = None,
    sketch_label: str = "Input Sketch",
    dpi: int = 150,
) -> Path:
    """Create a publication-quality banner: [Input Sketch | Variant 1 | Variant 2 | ...].

    Args:
        sketch: Input sketch (image or path).
        generated_images: Sequence of generated image variants.
        column_labels: Label for each generated image column.
        output_path: Destination path for saved PNG.
        title: Main chart title.
        subtitle: Optional subtitle describing parameters (e.g. Prompt, Seed, Backend).
        sketch_label: Title displayed over the sketch panel.
        dpi: Output resolution DPI (150 standard, 300 for print).

    Returns:
        Resolved absolute path to saved grid figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PIL images
    def _to_pil(img_in: str | Path | Image.Image) -> Image.Image:
        if isinstance(img_in, (str, Path)):
            return Image.open(img_in).convert("RGB")
        return img_in.convert("RGB")

    pil_sketch = _to_pil(sketch)
    pil_variants = [_to_pil(img) for img in generated_images]

    n_cols = 1 + len(pil_variants)
    fig_width = 3.6 * n_cols
    fig_height = 4.4 if subtitle else 4.0

    # Clean academic styling
    fig, axes = _subplots(1, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    ax_row = axes[0]

    # Panel 0: Input sketch with highlighted frame
    ax_row[0].imshow(pil_sketch)
    ax_row[0].set_title(
        sketch_label, fontsize=11, fontweight="bold", pad=8, color="#1a1a1a"
    )
    for spine in ax_row[0].spines.values():
        spine.set_edgecolor("#4a90e2")
        spine.set_linewidth(2.0)
    ax_row[0].set_xticks([])
    ax_row[0].set_yticks([])

    # Panels 1..N: Generated variants
    for idx, (img, label) in enumerate(
        zip(pil_variants, column_labels, strict=False), start=1
    ):
        ax = ax_row[idx]
        ax.imshow(img)
        ax.set_title(label, fontsize=10, pad=8, color="#222222")
        ax.axis("off")

    # Titles and Subtitles
    if title:
        y_pos = 1.05 if subtitle else 1.02
        fig.suptitle(title, fontsize=13, fontweight="bold", y=y_pos, color="#111111")
    if subtitle:
        fig.text(
            0.5,
            0.98,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
            color="#555555",
            wrap=True,
        )

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    logger.info("Saved experiment grid: %s", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# 2. 2D Parameter Matrix Grid (Rows x Columns)
# ---------------------------------------------------------------------------


def create_2d_matrix_grid(
    sketch: str | Path | Image.Image,
    matrix_images: list[list[str | Path | Image.Image]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_path: str | Path,
    title: str = "",
    subtitle: str | None = None,
    dpi: int = 150,
) -> Path:
    """Create a 2D parameter sweep matrix with row and column headers.

    Layout:
    [Sketch] [Col 1] [Col 2] ...
    [Row 1]  Img(0,0) Img(0,1)
    [Row 2]  Img(1,0) Img(1,1)

    Args:
        sketch: Input sketch.
        matrix_images: 2D list of shape (num_rows, num_cols).
        row_labels: Labels for each row.
        col_labels: Labels for each column.
        output_path: Destination path.
        title: Figure title.
        subtitle: Figure subtitle.
        dpi: DPI.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_pil(img_in: str | Path | Image.Image) -> Image.Image:
        if isinstance(img_in, (str, Path)):
            return Image.open(img_in).convert("RGB")
        return img_in.convert("RGB")

    pil_sketch = _to_pil(sketch)
    n_rows = len(row_labels)
    n_cols = len(col_labels)

    fig_w = 3.2 * (n_cols + 1)
    fig_h = 3.2 * n_rows + (0.8 if subtitle else 0.5)

    fig, axes = _subplots(n_rows, n_cols + 1, figsize=(fig_w, fig_h), squeeze=False)

    for r_idx in range(n_rows):
        # Column 0: Original sketch
        ax_sk = axes[r_idx, 0]
        ax_sk.imshow(pil_sketch)
        if r_idx == 0:
            ax_sk.set_title("Input Sketch", fontsize=10, fontweight="bold", pad=6)
        ax_sk.set_ylabel(row_labels[r_idx], fontsize=10, fontweight="bold", labelpad=8)
        ax_sk.set_xticks([])
        ax_sk.set_yticks([])

        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx + 1]
            img = _to_pil(matrix_images[r_idx][c_idx])
            ax.imshow(img)
            if r_idx == 0:
                ax.set_title(col_labels[c_idx], fontsize=10, pad=6)
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02, color="#111111")
    if subtitle:
        fig.text(
            0.5,
            0.98,
            subtitle,
            ha="center",
            fontsize=9,
            style="italic",
            color="#555555",
        )

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    logger.info("Saved 2D matrix grid: %s", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# 3. Hero Comparison Figure (Sketch -> Best Output Showcase)
# ---------------------------------------------------------------------------


def create_hero_figure(
    sketch_output_pairs: Sequence[dict[str, Any]],
    output_path: str | Path,
    title: str = "Bringing Sketches to Life with Generative AI",
    dpi: int = 150,
) -> Path:
    """Construct the main hero figure pairing each sketch with its premier output.

    Args:
        sketch_output_pairs: List of dicts, each containing:
            - ``sketch``: path or PIL Image.
            - ``output``: path or PIL Image.
            - ``caption``: descriptive caption (subject, preset, backend).
        output_path: Destination PNG file path.
        title: Main banner title.
        dpi: Output resolution.

    Returns:
        Resolved absolute path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = len(sketch_output_pairs)
    fig, axes = _subplots(n_pairs, 2, figsize=(8, 3.8 * n_pairs), squeeze=False)

    for i, pair in enumerate(sketch_output_pairs):
        sk_img = pair["sketch"]
        out_img = pair["output"]
        caption = pair.get("caption", f"Sketch {i+1}")

        if isinstance(sk_img, (str, Path)):
            sk_img = Image.open(sk_img).convert("RGB")
        if isinstance(out_img, (str, Path)):
            out_img = Image.open(out_img).convert("RGB")

        # Left: Sketch
        axes[i, 0].imshow(sk_img)
        axes[i, 0].set_title(
            f"Original Sketch #{i+1}", fontsize=11, fontweight="bold", pad=6
        )
        axes[i, 0].axis("off")

        # Right: Output
        axes[i, 1].imshow(out_img)
        axes[i, 1].set_title(
            f"Generated ({caption})", fontsize=11, fontweight="bold", pad=6
        )
        axes[i, 1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01, color="#111111")
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    logger.info("Saved hero figure: %s", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# 4. Standard 3-Panel Side-by-Side
# ---------------------------------------------------------------------------


def save_side_by_side(
    sketch: Image.Image,
    conditioning_map: Image.Image,
    generated: Image.Image,
    output_path: str | Path,
    titles: Sequence[str] = ("Sketch", "Conditioning Map", "Generated"),
    dpi: int = 150,
) -> Path:
    """Save a 3-panel figure: sketch | conditioning map | generated image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = _subplots(1, 3, figsize=(12, 4))
    images = [sketch, conditioning_map, generated]

    for ax, img, title in zip(axes, images, titles, strict=False):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    logger.debug("Saved side-by-side grid: %s", output_path)
    return output_path.resolve()
