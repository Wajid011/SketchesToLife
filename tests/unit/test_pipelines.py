"""Unit tests for sketch2image.pipelines helpers (no models loaded)."""

from __future__ import annotations

import types
from unittest import mock

from PIL import Image

from sketch2image.pipelines import _enable_vae_tiling, _match_adapter_channels


def _pipe_with_adapter(in_channels: int) -> types.SimpleNamespace:
    config = types.SimpleNamespace(in_channels=in_channels)
    return types.SimpleNamespace(adapter=types.SimpleNamespace(config=config))


def test_sketch_adapter_gets_grayscale() -> None:
    rgb = Image.new("RGB", (64, 64))
    assert _match_adapter_channels(_pipe_with_adapter(1), rgb).mode == "L"


def test_rgb_adapter_keeps_rgb() -> None:
    rgb = Image.new("RGB", (64, 64))
    assert _match_adapter_channels(_pipe_with_adapter(3), rgb).mode == "RGB"


def test_controlnet_pipeline_image_unchanged() -> None:
    rgb = Image.new("RGB", (64, 64))
    assert _match_adapter_channels(types.SimpleNamespace(), rgb) is rgb


class _NewDiffusersPipeline:
    """Recent diffusers: the pipeline no longer has enable_vae_tiling()."""

    def __init__(self) -> None:
        self.vae = mock.MagicMock()

    def __getattr__(self, name: str) -> None:
        raise AttributeError(name)


def test_vae_tiling_uses_vae_method_on_new_diffusers() -> None:
    pipe = _NewDiffusersPipeline()
    _enable_vae_tiling(pipe)
    pipe.vae.enable_tiling.assert_called_once()


def test_vae_tiling_on_old_diffusers_pipeline() -> None:
    pipe = mock.MagicMock()
    _enable_vae_tiling(pipe)
    pipe.vae.enable_tiling.assert_called_once()
    pipe.enable_vae_tiling.assert_not_called()


def test_vae_tiling_without_vae_is_a_no_op() -> None:
    pipe = mock.MagicMock(spec=[])  # no attributes at all
    _enable_vae_tiling(pipe)
