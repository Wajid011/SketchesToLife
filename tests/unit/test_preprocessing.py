"""Unit tests for sketch2image.preprocessing (CPU-safe)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from sketch2image.preprocessing import _resize_keep_aspect, load_sketch


# ---------------------------------------------------------------------------
# load_sketch
# ---------------------------------------------------------------------------


def test_load_sketch_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_sketch("/nonexistent/sketch.png")


def test_load_sketch_converts_to_rgb(tmp_path):
    # Create a small grayscale image
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
    # Longest edge should be ≤256 and both dims divisible by 8
    assert max(w, h) <= 256
    assert w % 8 == 0
    assert h % 8 == 0


# ---------------------------------------------------------------------------
# _resize_keep_aspect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orig_w,orig_h,max_side", [
    (1000, 500, 512),
    (200, 800, 256),
    (512, 512, 512),
])
def test_resize_keep_aspect_divisible_by_8(orig_w, orig_h, max_side):
    img = Image.new("RGB", (orig_w, orig_h))
    result = _resize_keep_aspect(img, max_side)
    w, h = result.size
    assert w % 8 == 0
    assert h % 8 == 0
    assert max(w, h) <= max_side + 7  # small rounding tolerance


# ---------------------------------------------------------------------------
# SketchPreprocessor — no-op mode (no controlnet_aux needed)
# ---------------------------------------------------------------------------


def test_preprocessor_none_mode():
    from sketch2image.preprocessing import SketchPreprocessor

    prep = SketchPreprocessor("none", resolution=512)
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    result = prep(img)
    assert result.mode == "RGB"
    assert result.size == (100, 100)


def test_preprocessor_missing_controlnet_aux():
    """Ensure a clear ImportError is raised when controlnet_aux is absent."""
    import sys
    from sketch2image.preprocessing import SketchPreprocessor

    prep = SketchPreprocessor("hed", resolution=512)

    # Simulate controlnet_aux not installed
    with patch.dict(sys.modules, {"controlnet_aux": None}):
        # _PREPROCESSOR_REGISTRY reset to trigger lazy load
        import sketch2image.preprocessing as _prep_mod
        _prep_mod._PREPROCESSOR_REGISTRY = {}
        with pytest.raises(ImportError, match="controlnet-aux"):
            prep._get_detector()
        # Restore for other tests
        _prep_mod._PREPROCESSOR_REGISTRY = {}
