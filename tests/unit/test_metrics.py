"""Unit tests for sketch2image.metrics (CPU-only, no model downloads)."""

from __future__ import annotations

import importlib.util
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from sketch2image.metrics import (
    ClipScorer,
    MetricConfig,
    chamfer_distance,
    edge_f1_iou,
    edge_ssim,
    evaluate_generation,
    generated_edges,
    reference_edges,
    structure_metrics,
    thin,
)


def _hline(y: int, size: int = 128, x0: int = 20, x1: int = 100) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    m[y, x0:x1] = 1
    return m


def _drawing(size: int = 512, offset: int = 0, stroke: int = 6) -> Image.Image:
    """Black rectangle + diagonal on white, like a scanned sketch."""
    import cv2

    canvas = np.full((size, size, 3), 255, np.uint8)
    cv2.rectangle(canvas, (100 + offset, 120), (400 + offset, 380), (0, 0, 0), stroke)
    cv2.line(canvas, (100 + offset, 120), (400 + offset, 380), (0, 0, 0), stroke)
    return Image.fromarray(canvas)


# ---------------------------------------------------------------------------
# F1 / IoU
# ---------------------------------------------------------------------------


def test_identical_maps_are_perfect() -> None:
    m = _hline(60)
    r = edge_f1_iou(m, m, tau=0)
    assert r["precision"] == r["recall"] == r["f1"] == pytest.approx(1.0)
    assert r["iou"] == pytest.approx(1.0)
    assert r["density"] == pytest.approx(1.0)


def test_tolerance_absorbs_small_shift() -> None:
    a, b = _hline(60), _hline(63)
    assert edge_f1_iou(a, b, tau=0)["f1"] == 0.0
    assert edge_f1_iou(a, b, tau=3)["f1"] == pytest.approx(1.0)


def test_extra_edges_lower_precision_not_recall() -> None:
    ref = _hline(30)
    gen = ref | _hline(90)  # sketch line kept + invented line
    r = edge_f1_iou(ref, gen, tau=2)
    assert r["recall"] == pytest.approx(1.0)
    assert r["precision"] == pytest.approx(0.5)
    assert r["density"] == pytest.approx(2.0)


def test_missing_edges_lower_recall() -> None:
    ref = _hline(30) | _hline(90)
    r = edge_f1_iou(ref, _hline(30), tau=2)
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(0.5)


def test_empty_generated_map_scores_zero() -> None:
    r = edge_f1_iou(_hline(30), np.zeros((128, 128), np.uint8))
    assert r["f1"] == 0.0 and r["iou"] == 0.0


# ---------------------------------------------------------------------------
# Chamfer / SSIM
# ---------------------------------------------------------------------------


def test_chamfer_equals_shift_distance() -> None:
    r = chamfer_distance(_hline(50), _hline(60))
    assert r["chamfer"] == pytest.approx(10.0)
    assert r["chamfer_s2g"] == pytest.approx(10.0)


def test_chamfer_directions_separate_missing_from_invented() -> None:
    ref = _hline(30)
    gen = ref | _hline(100)
    r = chamfer_distance(ref, gen)
    assert r["chamfer_s2g"] == pytest.approx(0.0)  # nothing missing
    assert r["chamfer_g2s"] > 30  # invented line far away


def test_chamfer_empty_is_diagonal() -> None:
    r = chamfer_distance(_hline(30), np.zeros((128, 128), np.uint8))
    assert r["chamfer"] == pytest.approx(np.hypot(128, 128))


def test_edge_ssim_ordering() -> None:
    a = _hline(60)
    assert edge_ssim(a, a) == pytest.approx(1.0)
    assert edge_ssim(a, _hline(62)) > edge_ssim(a, _hline(110))


def test_structure_metrics_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        structure_metrics(np.zeros((10, 10), np.uint8), np.zeros((12, 12), np.uint8))


# ---------------------------------------------------------------------------
# Edge extraction on images
# ---------------------------------------------------------------------------


def test_thin_produces_one_pixel_lines() -> None:
    thick = np.zeros((64, 64), np.uint8)
    thick[30:37, 5:60] = 1
    skel = thin(thick)
    assert skel.sum() < thick.sum() / 4
    assert skel[:, 32].sum() == 1


def test_reference_edges_frame_and_size() -> None:
    ref = reference_edges(_drawing(), resolution=1024, eval_resolution=512)
    assert ref.shape == (512, 512)
    assert ref.sum() > 0


def test_canny_on_rendered_sketch_recovers_structure() -> None:
    """The generator 'rendering' the exact sketch must score near-perfect recall."""
    sketch = _drawing()
    ref = reference_edges(sketch, resolution=512)
    gen = generated_edges(sketch, "canny")
    r = structure_metrics(ref, gen, tau=4)
    assert r["recall"] > 0.9
    # Canny finds both borders of a 6-px stroke, ~3 px from the sketch centre line,
    # so Chamfer has a floor of ~half the stroke width.
    assert r["chamfer"] < 4.0


def test_shifted_rendering_scores_worse() -> None:
    sketch = _drawing()
    ref = reference_edges(sketch, resolution=512)
    near = structure_metrics(ref, generated_edges(_drawing(offset=2), "canny"))
    far = structure_metrics(ref, generated_edges(_drawing(offset=40), "canny"))
    assert near["f1"] > far["f1"]
    assert near["chamfer"] < far["chamfer"]


def test_unknown_edge_method() -> None:
    with pytest.raises(ValueError):
        generated_edges(_drawing(), "sobel")


@pytest.mark.skipif(
    importlib.util.find_spec("controlnet_aux") is None,
    reason="controlnet_aux not installed (GPU env only)",
)
def test_hed_detector_called_with_eval_resolution() -> None:
    fake = mock.MagicMock(return_value=Image.new("L", (512, 512), 255))
    with mock.patch("sketch2image.preprocessing._get_aux_detector", return_value=fake):
        out = generated_edges(_drawing(), "hed")
    assert out.shape == (512, 512)
    assert fake.call_args.kwargs["detect_resolution"] == 512


# ---------------------------------------------------------------------------
# CLIP (mocked - never downloads weights)
# ---------------------------------------------------------------------------


def test_clip_score_formula_with_mocked_model() -> None:
    torch = pytest.importorskip("torch")
    scorer = ClipScorer(device="cpu")
    out = mock.MagicMock()
    out.image_embeds = torch.tensor([[3.0, 4.0]])  # unit: (0.6, 0.8)
    out.text_embeds = torch.tensor([[1.0, 0.0]])
    scorer._model = mock.MagicMock(return_value=out)
    scorer._processor = mock.MagicMock()
    assert scorer.score(_drawing(64), "a box") == pytest.approx(60.0)

    out.text_embeds = torch.tensor([[-1.0, 0.0]])  # negative cosine clamps to 0
    assert scorer.score(_drawing(64), "a box") == 0.0


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------


def test_evaluate_generation_keys_without_clip() -> None:
    res = evaluate_generation(
        _drawing(),
        _drawing(offset=1),
        resolution=512,
        config=MetricConfig(edge_methods=("canny",)),
    )
    for key in (
        "f1",
        "recall",
        "precision",
        "iou",
        "chamfer",
        "chamfer_s2g",
        "chamfer_g2s",
        "ssim",
        "density",
    ):
        assert f"canny_{key}" in res
    assert not any(k.startswith("clip") for k in res)


def test_evaluate_generation_scores_each_prompt() -> None:
    clip = mock.MagicMock(spec=ClipScorer)
    clip.score.side_effect = [30.0, 25.0]
    res = evaluate_generation(
        _drawing(),
        _drawing(),
        512,
        prompts={"": "a box", "_full": "a box, watercolor"},
        clip=clip,
        config=MetricConfig(edge_methods=("canny",)),
    )
    assert res["clip_score"] == 30.0 and res["clip_score_full"] == 25.0
