"""Unit tests for sketch2image.generate and SketchToImageGenerator (CPU-safe)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from sketch2image.generate import (
    DEFAULT_NEGATIVE_PROMPT,
    STYLE_PRESETS,
    apply_style_preset,
    generate_for_sketch,
)
from sketch2image.pipelines import (
    DEFAULT_MODEL_IDS,
    SketchToImageGenerator,
    configure_scheduler,
)


# ---------------------------------------------------------------------------
# 1. Style Presets
# ---------------------------------------------------------------------------


def test_apply_style_preset_none():
    prompt = "a mountain cabin"
    styled, negative = apply_style_preset(prompt, preset=None)
    assert styled == "a mountain cabin"
    assert negative == DEFAULT_NEGATIVE_PROMPT


def test_apply_style_preset_photorealistic():
    prompt = "a sports car"
    styled, negative = apply_style_preset(prompt, preset="photorealistic")
    assert "8k UHD" in styled
    assert "cartoon" in negative


def test_apply_style_preset_custom_negative():
    prompt = "a cat"
    styled, negative = apply_style_preset(prompt, preset="cyberpunk", custom_negative="blur, dark")
    assert "neon" in styled
    assert "blur, dark" in negative


# ---------------------------------------------------------------------------
# 2. Generator Initialization & Backend Registry
# ---------------------------------------------------------------------------


def test_generator_init_defaults():
    gen = SketchToImageGenerator(mock=True)
    assert gen.backend == "sd15_controlnet"
    assert gen.controlnet_type == "scribble"
    assert gen.mock is True


def test_generator_cache_key():
    gen = SketchToImageGenerator(backend="sdxl_controlnet", controlnet_type="scribble", mock=True)
    key = gen._get_cache_key()
    assert "sdxl_controlnet" in key


def test_generator_invalid_backend():
    gen = SketchToImageGenerator(backend="nonexistent_backend", mock=False)
    with patch.object(gen, "_is_torch_cuda_available", return_value=True):
        with pytest.raises(ValueError, match="Unknown backend"):
            gen.load()


# ---------------------------------------------------------------------------
# 3. Generation & JSON Sidecar
# ---------------------------------------------------------------------------


def test_generate_mock_output_and_sidecar(tmp_path: Path):
    # Create a small dummy sketch
    sketch_file = tmp_path / "test_sketch.png"
    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    img.save(sketch_file)

    gen = SketchToImageGenerator(mock=True)
    out_dir = tmp_path / "outputs"

    results = generate_for_sketch(
        sketch_path=sketch_file,
        prompt="a cozy modern house",
        generator=gen,
        output_dir=out_dir,
        preset="photorealistic",
        seed=101,
        num_images_per_prompt=1,
    )

    assert len(results) == 1
    res = results[0]
    assert res["image_path"].exists()
    assert res["sidecar"].exists()
    assert res["grid_path"].exists()

    # Verify JSON sidecar content
    with res["sidecar"].open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["raw_prompt"] == "a cozy modern house"
    assert meta["style_preset"] == "photorealistic"
    assert meta["seed"] == 101
    assert "runtime_sec" in meta
    assert "peak_vram_mb" in meta
    assert "model_ids" in meta


def test_generate_missing_sketch_raises(tmp_path: Path):
    gen = SketchToImageGenerator(mock=True)
    with pytest.raises(FileNotFoundError):
        generate_for_sketch(
            sketch_path=tmp_path / "nonexistent.png",
            prompt="test",
            generator=gen,
            output_dir=tmp_path,
        )
