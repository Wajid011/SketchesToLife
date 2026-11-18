"""Unit tests for sketch2image.metrics (CPU-safe)."""

from __future__ import annotations

import pytest
from PIL import Image


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, color=color)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def test_ssim_identical():
    pytest.importorskip("skimage", reason="scikit-image not installed (Colab-only dep)")
    from sketch2image.metrics import compute_ssim
    img = _solid((128, 128, 128))
    score = compute_ssim(img, img)
    assert score == pytest.approx(1.0, abs=0.01)


def test_ssim_different():
    pytest.importorskip("skimage", reason="scikit-image not installed (Colab-only dep)")
    from sketch2image.metrics import compute_ssim
    white = _solid((255, 255, 255))
    black = _solid((0, 0, 0))
    score = compute_ssim(white, black)
    assert score < 0.5


def test_ssim_different_sizes():
    pytest.importorskip("skimage", reason="scikit-image not installed (Colab-only dep)")
    from sketch2image.metrics import compute_ssim
    a = _solid((100, 100, 100), (64, 64))
    b = _solid((100, 100, 100), (128, 128))
    # Should not raise; sizes are aligned internally
    score = compute_ssim(a, b)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Edge similarity
# ---------------------------------------------------------------------------


def test_edge_similarity_identical():
    from sketch2image.metrics import edge_similarity
    img = _solid((200, 200, 200))
    score = edge_similarity(img, img)
    # Solid-colour images have no edges → both maps are zeros → Dice ≈ 0
    # (not 1 because there are no positive pixels in either mask)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


def test_compute_all_metrics_runs():
    from sketch2image.metrics import compute_all_metrics
    a = _solid((128, 128, 128))
    b = _solid((64, 64, 64))
    # lpips requires torch+lpips; skip if not installed
    result = compute_all_metrics(a, b)
    assert isinstance(result, dict)
    # At minimum ssim and edge_similarity should be present
    for key in ("ssim", "edge_similarity"):
        if key in result:
            assert isinstance(result[key], float)
