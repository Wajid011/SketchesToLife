"""Unit tests for sketch2image.preprocessing (CPU-safe).

Tests:
- Automatic inversion detection (paper white-to-black inversion)
- Resize and letterbox padding (multiples of 8/64, aspect ratio, padding)
- Cleaning operations: small-blob removal, line dilation/erosion
- Output shape and color channels for prepare_condition
- Error handling for missing dependencies and invalid modes
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from sketch2image.preprocessing import (
    SketchPreprocessor,
    _resize_keep_aspect,
    adjust_line_thickness,
    auto_invert_to_white_lines,
    clean_sketch,
    extract_canny,
    load_and_orient_image,
    load_sketch,
    normalize_illumination,
    prepare_condition,
    remove_small_blobs,
    resize_and_pad,
)


# ---------------------------------------------------------------------------
# 1. Automatic Inversion Detection
# ---------------------------------------------------------------------------


def test_auto_invert_black_lines_on_white_paper():
    """Typical sketch: white paper background (255) with black pencil lines (0)."""
    # White paper canvas with a black line in the middle
    canvas = np.full((100, 100), 255, dtype=np.uint8)
    canvas[50, 20:80] = 0

    inverted, was_inverted = auto_invert_to_white_lines(canvas)
    assert was_inverted is True
    # Inverted result should have black background (0) and white line (255)
    assert inverted[0, 0] == 0
    assert inverted[50, 50] == 255


def test_auto_invert_white_lines_on_black_background():
    """Already inverted sketch: black background (0) with white strokes (255)."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    canvas[50, 20:80] = 255

    result, was_inverted = auto_invert_to_white_lines(canvas)
    assert was_inverted is False
    assert result[0, 0] == 0
    assert result[50, 50] == 255


# ---------------------------------------------------------------------------
# 2. Resize and Padding (Multiple of 8 / 64)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_res", [512, 1024])
def test_resize_and_pad_square_sd15_sdxl(target_res: int):
    """Output shape must exactly match target resolution and be divisible by 64."""
    img = np.zeros((300, 700), dtype=np.uint8)
    padded = resize_and_pad(img, resolution=target_res, multiple=64)

    assert padded.shape == (target_res, target_res)
    assert padded.shape[0] % 64 == 0
    assert padded.shape[1] % 64 == 0


def test_resize_and_pad_aspect_ratio_landscape():
    """Landscape image padded symmetrically vertically with 0s."""
    # 100 high x 200 wide -> aspect ratio 2:1
    img = np.full((100, 200), 255, dtype=np.uint8)
    padded = resize_and_pad(img, resolution=512, multiple=64)

    assert padded.shape == (512, 512)
    # Scaled to 512 width, height should be 256
    # Padding top and bottom should be (512 - 256) // 2 = 128
    assert padded[10, 256] == 0       # Top padding area
    assert padded[500, 256] == 0      # Bottom padding area
    assert padded[256, 256] == 255    # Center content area


def test_resize_and_pad_aspect_ratio_portrait():
    """Portrait image padded symmetrically horizontally with 0s."""
    # 200 high x 100 wide -> aspect ratio 1:2
    img = np.full((200, 100), 255, dtype=np.uint8)
    padded = resize_and_pad(img, resolution=512, multiple=64)

    assert padded.shape == (512, 512)
    assert padded[256, 10] == 0       # Left padding area
    assert padded[256, 500] == 0      # Right padding area
    assert padded[256, 256] == 255    # Center content area


def test_resize_and_pad_rgb_3channel():
    """Ensure 3D arrays retain channel dimension."""
    img = np.ones((150, 300, 3), dtype=np.uint8) * 120
    padded = resize_and_pad(img, resolution=512, multiple=64)
    assert padded.shape == (512, 512, 3)


# ---------------------------------------------------------------------------
# 3. Small-Blob Removal & Line Thickness
# ---------------------------------------------------------------------------


def test_remove_small_blobs():
    """Speckles smaller than min_area must be removed, larger strokes retained."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    # Tiny 2x2 speckle (area 4)
    canvas[10:12, 10:12] = 255
    # Large rectangle 10x10 (area 100)
    canvas[40:50, 40:50] = 255

    cleaned = remove_small_blobs(canvas, min_area=25)
    assert np.all(cleaned[10:12, 10:12] == 0)   # Speckle removed
    assert np.all(cleaned[40:50, 40:50] == 255) # Stroke preserved


def test_adjust_line_thickness():
    """Dilation increases foreground pixels, erosion decreases them."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    canvas[45:55, 45:55] = 255  # 10x10 square = 100 px
    base_count = np.count_nonzero(canvas)

    dilated = adjust_line_thickness(canvas, line_thickness=1)
    dilated_count = np.count_nonzero(dilated)
    assert dilated_count > base_count

    eroded = adjust_line_thickness(canvas, line_thickness=-1)
    eroded_count = np.count_nonzero(eroded)
    assert eroded_count < base_count


# ---------------------------------------------------------------------------
# 4. prepare_condition Output Shapes & Modes
# ---------------------------------------------------------------------------


def test_prepare_condition_clean_mode():
    """'clean' mode must produce an RGB PIL Image of exact target size."""
    raw = Image.new("RGB", (200, 400), color=(255, 255, 255))
    result = prepare_condition(raw, mode="clean", resolution=512)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (512, 512)


def test_prepare_condition_canny_mode():
    """'canny' mode must produce an RGB PIL Image of exact target size."""
    raw = Image.new("RGB", (300, 200), color=(100, 100, 100))
    result = prepare_condition(raw, mode="canny", resolution=512)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (512, 512)


def test_prepare_condition_sdxl_resolution():
    """SDXL resolution (1024) must yield (1024, 1024) RGB image."""
    raw = Image.new("RGB", (400, 300), color=(240, 240, 240))
    result = prepare_condition(raw, mode="clean", resolution=1024)
    assert result.size == (1024, 1024)


def test_prepare_condition_invalid_mode():
    """Invalid mode name should raise ValueError."""
    raw = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError, match="Unknown mode"):
        prepare_condition(raw, mode="unsupported_mode")


# ---------------------------------------------------------------------------
# 5. Legacy & Compatibility Tests
# ---------------------------------------------------------------------------


def test_load_sketch_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_sketch("/nonexistent/sketch.png")


def test_load_sketch_converts_to_rgb(tmp_path):
    img = Image.new("L", (100, 80), color=128)
    p = tmp_path / "sketch.png"
    img.save(p)

    result = load_sketch(p)
    assert result.mode == "RGB"
    assert result.size == (100, 80)


def test_load_sketch_resize(tmp_path):
    img = Image.new("RGB", (1000, 500))
    p = tmp_path / "big.png"
    img.save(p)

    result = load_sketch(p, resolution=256)
    w, h = result.size
    assert max(w, h) == 256
    assert w % 64 == 0
    assert h % 64 == 0


def test_preprocessor_none_mode():
    prep = SketchPreprocessor("none", resolution=512)
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    result = prep(img)
    assert result.mode == "RGB"
    assert result.size == (512, 512)


def test_preprocessor_missing_controlnet_aux():
    """Ensure clear error when controlnet_aux is not installed."""
    import sys
    import sketch2image.preprocessing as prep_mod

    # Simulate controlnet_aux absent
    with patch.dict(sys.modules, {"controlnet_aux": None}):
        prep_mod._AUX_DETECTOR_REGISTRY = {}
        with pytest.raises(ImportError, match="controlnet-aux"):
            prep_mod._get_aux_detector("hed")
        prep_mod._AUX_DETECTOR_REGISTRY = {}
