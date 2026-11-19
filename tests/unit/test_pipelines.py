"""Unit tests for sketch2image.pipelines helpers (no models loaded)."""

from __future__ import annotations

from unittest import mock

from sketch2image.pipelines import _enable_vae_tiling


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
