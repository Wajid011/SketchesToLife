"""Unit tests for sketch2image.visualize and experiment configs (CPU-safe)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from sketch2image.visualize import (
    create_2d_matrix_grid,
    create_experiment_grid,
    create_hero_figure,
    save_side_by_side,
)


def _make_dummy_image(color=(128, 128, 128), size=(64, 64)) -> Image.Image:
    return Image.new("RGB", size, color=color)


# ---------------------------------------------------------------------------
# 1. Visualization Grids
# ---------------------------------------------------------------------------


def test_create_experiment_grid(tmp_path: Path):
    sketch = _make_dummy_image((255, 255, 255))
    v1 = _make_dummy_image((200, 50, 50))
    v2 = _make_dummy_image((50, 200, 50))
    out_file = tmp_path / "test_exp_grid.png"

    res = create_experiment_grid(
        sketch=sketch,
        generated_images=[v1, v2],
        column_labels=["Variant A", "Variant B"],
        output_path=out_file,
        title="Test Experiment",
        subtitle="Testing subtitle",
    )

    assert res.exists()
    assert out_file.stat().st_size > 0


def test_create_2d_matrix_grid(tmp_path: Path):
    sketch = _make_dummy_image((255, 255, 255))
    row1 = [_make_dummy_image((10, 10, 10)), _make_dummy_image((20, 20, 20))]
    row2 = [_make_dummy_image((30, 30, 30)), _make_dummy_image((40, 40, 40))]
    out_file = tmp_path / "test_matrix.png"

    res = create_2d_matrix_grid(
        sketch=sketch,
        matrix_images=[row1, row2],
        row_labels=["Row 1", "Row 2"],
        col_labels=["Col A", "Col B"],
        output_path=out_file,
        title="Test 2D Matrix",
    )

    assert res.exists()
    assert out_file.stat().st_size > 0


def test_create_hero_figure(tmp_path: Path):
    sk1 = _make_dummy_image((240, 240, 240))
    out1 = _make_dummy_image((100, 150, 200))
    sk2 = _make_dummy_image((240, 240, 240))
    out2 = _make_dummy_image((200, 150, 100))

    pairs = [
        {"sketch": sk1, "output": out1, "caption": "Scene 1 — Photorealistic"},
        {"sketch": sk2, "output": out2, "caption": "Scene 2 — Concept Art"},
    ]
    out_file = tmp_path / "test_hero.png"

    res = create_hero_figure(pairs, output_path=out_file)
    assert res.exists()
    assert out_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# 2. Experiment Configurations Validation
# ---------------------------------------------------------------------------


def test_all_experiment_yamls_valid():
    """Verify all 8 experiment configs parse and contain required schema keys."""
    exp_dir = Path("configs/experiments")
    exp_files = sorted(exp_dir.glob("*.yaml"))
    assert len(exp_files) == 8, f"Expected 8 experiment YAMLs, found {len(exp_files)}"

    for f in exp_files:
        with f.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

        assert "id" in data, f"Missing 'id' in {f.name}"
        assert "title" in data, f"Missing 'title' in {f.name}"
        assert "sweep" in data, f"Missing 'sweep' in {f.name}"
        assert "type" in data["sweep"], f"Missing 'sweep.type' in {f.name}"
