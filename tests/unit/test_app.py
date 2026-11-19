"""Tests for app/app.py helpers (skipped when gradio is not installed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("gradio")

_spec = importlib.util.spec_from_file_location(
    "s2l_app", Path(__file__).resolve().parents[2] / "app" / "app.py"
)
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


def _transparent_drawing() -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(10, 54):
        img.putpixel((x, 32), (0, 0, 0, 255))
    return img


def test_flatten_on_white_keeps_black_strokes_visible() -> None:
    rgb = np.asarray(app.flatten_on_white(_transparent_drawing()))
    assert rgb[0, 0].tolist() == [255, 255, 255]  # transparent -> white, not black
    assert rgb[32, 20].tolist() == [0, 0, 0]


def test_resolve_sketch_blank_canvas_is_none() -> None:
    blank = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    assert app.resolve_sketch("draw", None, {"composite": blank}) is None
    assert app.resolve_sketch("upload", None, None) is None


def test_resolve_sketch_picks_active_source() -> None:
    drawn = _transparent_drawing()
    up = Image.new("RGB", (64, 64), "white")
    up.putpixel((5, 5), (0, 0, 0))
    assert app.resolve_sketch("draw", up, drawn).size == (64, 64)
    assert np.asarray(app.resolve_sketch("upload", up, drawn))[5, 5].tolist() == [
        0,
        0,
        0,
    ]


def test_oom_detection_and_message() -> None:
    assert app._is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not app._is_oom(ValueError("bad prompt"))
    msg = app._oom_message("sdxl_controlnet", 3)
    assert "1 image" in msg and "SD 1.5" in msg


def test_generate_mock_flow_writes_zip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app, "APP_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app, "MODELS", app.ModelManager(force_mock=True))
    sketch = Image.new("RGB", (128, 96), "white")
    for x in range(20, 100):
        sketch.putpixel((x, 48), (0, 0, 0))
        sketch.putpixel((x, 49), (0, 0, 0))
    gallery, side, summary, zip_path, seed = app.generate(
        "upload",
        sketch,
        None,
        "a line",
        "watercolor",
        "",
        list(app.BACKENDS)[0],
        "scribble",
        1.0,
        7.5,
        20,
        1.0,
        7,
        False,
        2,
        list(app.PREPROCESS_MODES)[0],
        0,
        progress=lambda *a, **k: None,
    )
    assert len(gallery) == 2 and seed == 7
    assert "Mock mode" in summary
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        assert {
            "settings.json",
            "result_0.png",
            "result_1.png",
            "control_map.png",
        } <= set(zf.namelist())
