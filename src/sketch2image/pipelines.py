"""Diffusion pipeline wrappers for sketch-to-image generation.

Three backends are supported, each implemented as a thin wrapper around
the Hugging Face ``diffusers`` library:

- :class:`SD15ControlNetPipeline`  — SD 1.5 + ControlNet (primary, T4-friendly)
- :class:`SDXLControlNetPipeline`  — SDXL + ControlNet  (comparison, needs ≥16 GB)
- :class:`T2IAdapterPipeline`      — T2I-Adapter + SD 1.5 (lightweight comparison)

All pipelines share:
- ``torch_dtype=float16`` by default.
- Attention slicing enabled always.
- ``enable_model_cpu_offload()`` available for SDXL.
- Fixed-seed generation via ``torch.Generator``.

This module deliberately contains **no** top-level diffusers imports so that
local unit tests (CPU, no diffusers) can import this file without errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _BaseSketchPipeline:
    """Abstract base for all sketch-to-image pipelines.

    Args:
        config: Merged experiment configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._pipe = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights into memory.

        Must be called before :meth:`generate`. Idempotent — safe to call
        multiple times; subsequent calls are no-ops.
        """
        raise NotImplementedError

    def generate(
        self,
        conditioning_map: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        **kwargs: Any,
    ) -> Image.Image:
        """Generate an image from a conditioning map and text prompt.

        Args:
            conditioning_map: Preprocessed edge/sketch map (RGB PIL Image).
            prompt: Positive text prompt.
            negative_prompt: Negative text prompt.
            seed: RNG seed for reproducibility.
            **kwargs: Additional pipeline-specific keyword arguments that
                override values from ``self.config``.

        Returns:
            Generated image as a PIL Image.
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Release GPU memory by deleting the pipeline object."""
        import gc

        self._pipe = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.info("Pipeline unloaded and GPU cache cleared.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_generator(self, seed: int) -> Any:
        """Create a seeded ``torch.Generator`` for the configured device.

        Args:
            seed: Integer seed value.

        Returns:
            Seeded ``torch.Generator`` instance.
        """
        import torch

        device = self.config.get("device", "cuda")
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return gen

    def _dtype(self) -> Any:
        """Return the configured ``torch.dtype``.

        Returns:
            ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``.
        """
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.get("dtype", "float16"), torch.float16)


# ---------------------------------------------------------------------------
# SD 1.5 + ControlNet  (primary backend)
# ---------------------------------------------------------------------------


class SD15ControlNetPipeline(_BaseSketchPipeline):
    """Stable Diffusion 1.5 + ControlNet inference pipeline.

    Primary backend — designed to run on a free Colab T4 (16 GB VRAM)
    with ``float16`` and attention slicing.

    Args:
        config: Merged experiment config. Required keys:
            ``models.sd15_base``, ``models.controlnet_<type>``,
            ``controlnet_type``, ``device``, ``dtype``,
            ``enable_attention_slicing``, ``scheduler``.
    """

    def load(self) -> None:
        """Load SD 1.5 UNet + ControlNet weights from Hugging Face."""
        if self._pipe is not None:
            logger.debug("SD15ControlNetPipeline already loaded; skipping.")
            return

        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        controlnet_id = self.config["models"][
            f"controlnet_{self.config['controlnet_type']}"
        ]
        base_id = self.config["models"]["sd15_base"]
        device = self.config.get("device", "cuda")
        dtype = self._dtype()

        logger.info("Loading ControlNet from '%s' …", controlnet_id)
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)

        logger.info("Loading SD 1.5 base from '%s' …", base_id)
        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_id,
            controlnet=controlnet,
            torch_dtype=dtype,
        )
        self._pipe = self._pipe.to(device)

        # Memory optimisations
        if self.config.get("enable_attention_slicing", True):
            self._pipe.enable_attention_slicing()
            logger.debug("Attention slicing enabled.")

        if self.config.get("enable_xformers", False):
            self._pipe.enable_xformers_memory_efficient_attention()
            logger.debug("xFormers memory-efficient attention enabled.")

        self._set_scheduler()
        logger.info("SD15ControlNetPipeline ready on '%s'.", device)

    def generate(
        self,
        conditioning_map: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        **kwargs: Any,
    ) -> Image.Image:
        """Run a single inference pass.

        Args:
            conditioning_map: Preprocessed sketch edge map.
            prompt: Positive text prompt.
            negative_prompt: Negative text prompt.
            seed: Reproducibility seed.
            **kwargs: Overrides for ``num_inference_steps``,
                ``guidance_scale``, ``controlnet_conditioning_scale``,
                ``control_guidance_start``, ``control_guidance_end``,
                ``resolution``.

        Returns:
            Generated image (PIL Image).
        """
        if self._pipe is None:
            raise RuntimeError("Call load() before generate().")

        cfg = self.config
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=conditioning_map,
            num_inference_steps=kwargs.get("num_inference_steps", cfg["num_inference_steps"]),
            guidance_scale=kwargs.get("guidance_scale", cfg["guidance_scale"]),
            controlnet_conditioning_scale=float(
                kwargs.get(
                    "controlnet_conditioning_scale",
                    cfg["controlnet_conditioning_scale"],
                )
            ),
            control_guidance_start=kwargs.get(
                "control_guidance_start", cfg.get("control_guidance_start", 0.0)
            ),
            control_guidance_end=kwargs.get(
                "control_guidance_end", cfg.get("control_guidance_end", 1.0)
            ),
            height=kwargs.get("resolution", cfg["resolution"]),
            width=kwargs.get("resolution", cfg["resolution"]),
            generator=self._make_generator(seed),
        )
        return result.images[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_scheduler(self) -> None:
        """Replace the pipeline's scheduler based on config."""
        scheduler_name = self.config.get("scheduler", "DPMSolverMultistep")
        scheduler_map = {
            "DDIM": "diffusers.DDIMScheduler",
            "DPMSolverMultistep": "diffusers.DPMSolverMultistepScheduler",
            "UniPCMultistep": "diffusers.UniPCMultistepScheduler",
            "PNDM": "diffusers.PNDMScheduler",
            "Euler": "diffusers.EulerDiscreteScheduler",
            "EulerAncestral": "diffusers.EulerAncestralDiscreteScheduler",
        }
        if scheduler_name not in scheduler_map:
            logger.warning("Unknown scheduler '%s'; keeping default.", scheduler_name)
            return

        import importlib

        module_path, cls_name = scheduler_map[scheduler_name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        scheduler_cls = getattr(module, cls_name)
        self._pipe.scheduler = scheduler_cls.from_config(self._pipe.scheduler.config)
        logger.debug("Scheduler set to '%s'.", scheduler_name)


# ---------------------------------------------------------------------------
# SDXL + ControlNet  (comparison backend)
# ---------------------------------------------------------------------------


class SDXLControlNetPipeline(_BaseSketchPipeline):
    """Stable Diffusion XL + ControlNet inference pipeline.

    Comparison backend — produces 1024×1024 images but requires ≥16 GB VRAM.
    Automatically enables ``enable_model_cpu_offload`` to fit on a T4.

    Args:
        config: Merged experiment config. Required keys as for
            :class:`SD15ControlNetPipeline`, plus ``models.sdxl_base``
            and ``models.controlnet_sdxl_scribble``.
    """

    def load(self) -> None:
        """Load SDXL UNet + ControlNet weights from Hugging Face."""
        if self._pipe is not None:
            return

        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        controlnet_id = self.config["models"]["controlnet_sdxl_scribble"]
        base_id = self.config["models"]["sdxl_base"]
        dtype = self._dtype()

        logger.info("Loading SDXL ControlNet from '%s' …", controlnet_id)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=dtype, use_safetensors=True
        )

        logger.info("Loading SDXL base from '%s' …", base_id)
        self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        # CPU offload is mandatory for SDXL on T4 (12 GB model, 16 GB VRAM)
        self._pipe.enable_model_cpu_offload()
        logger.info("SDXL pipeline loaded with model CPU offload.")

    def generate(
        self,
        conditioning_map: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        **kwargs: Any,
    ) -> Image.Image:
        """Run SDXL inference.

        Args:
            conditioning_map: Preprocessed sketch edge map.
            prompt: Positive text prompt.
            negative_prompt: Negative text prompt.
            seed: Reproducibility seed.
            **kwargs: Same overrides as :class:`SD15ControlNetPipeline`.

        Returns:
            Generated image at 1024×1024 (PIL Image).
        """
        if self._pipe is None:
            raise RuntimeError("Call load() before generate().")

        cfg = self.config
        resolution = kwargs.get("resolution", cfg.get("resolution", 1024))

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=conditioning_map,
            num_inference_steps=kwargs.get("num_inference_steps", cfg["num_inference_steps"]),
            guidance_scale=kwargs.get("guidance_scale", cfg["guidance_scale"]),
            controlnet_conditioning_scale=float(
                kwargs.get(
                    "controlnet_conditioning_scale",
                    cfg["controlnet_conditioning_scale"],
                )
            ),
            height=resolution,
            width=resolution,
            generator=self._make_generator(seed),
        )
        return result.images[0]


# ---------------------------------------------------------------------------
# T2I-Adapter  (lightweight comparison)
# ---------------------------------------------------------------------------


class T2IAdapterPipeline(_BaseSketchPipeline):
    """T2I-Adapter + SD 1.5 inference pipeline.

    Lightweight alternative to ControlNet — the adapter adds only ~300 MB
    over the SD 1.5 base model, making it very memory-efficient.

    Args:
        config: Merged experiment config. Required keys:
            ``models.sd15_base``, ``models.t2i_adapter_sketch``,
            ``device``, ``dtype``.
    """

    def load(self) -> None:
        """Load T2I-Adapter weights from Hugging Face."""
        if self._pipe is not None:
            return

        from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

        adapter_id = self.config["models"]["t2i_adapter_sketch"]
        base_id = self.config["models"]["sd15_base"]
        device = self.config.get("device", "cuda")
        dtype = self._dtype()

        logger.info("Loading T2I-Adapter from '%s' …", adapter_id)
        adapter = T2IAdapter.from_pretrained(adapter_id, torch_dtype=dtype)

        logger.info("Loading SD 1.5 base from '%s' …", base_id)
        self._pipe = StableDiffusionAdapterPipeline.from_pretrained(
            base_id,
            adapter=adapter,
            torch_dtype=dtype,
        )
        self._pipe = self._pipe.to(device)

        if self.config.get("enable_attention_slicing", True):
            self._pipe.enable_attention_slicing()

        logger.info("T2IAdapterPipeline ready on '%s'.", device)

    def generate(
        self,
        conditioning_map: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        **kwargs: Any,
    ) -> Image.Image:
        """Run T2I-Adapter inference.

        Args:
            conditioning_map: Sketch conditioning image.
            prompt: Positive text prompt.
            negative_prompt: Negative text prompt.
            seed: Reproducibility seed.
            **kwargs: Overrides for ``num_inference_steps``,
                ``guidance_scale``, ``resolution``.

        Returns:
            Generated image (PIL Image).
        """
        if self._pipe is None:
            raise RuntimeError("Call load() before generate().")

        cfg = self.config
        resolution = kwargs.get("resolution", cfg["resolution"])

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=conditioning_map,
            num_inference_steps=kwargs.get("num_inference_steps", cfg["num_inference_steps"]),
            guidance_scale=kwargs.get("guidance_scale", cfg["guidance_scale"]),
            height=resolution,
            width=resolution,
            generator=self._make_generator(seed),
        )
        return result.images[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_pipeline(config: dict[str, Any]) -> _BaseSketchPipeline:
    """Instantiate and return the pipeline specified by ``config['backend']``.

    Args:
        config: Merged experiment configuration dictionary.

    Returns:
        An unloaded pipeline instance. Call ``.load()`` before ``.generate()``.

    Raises:
        ValueError: If ``config['backend']`` is not recognised.
    """
    backend = config.get("backend", "sd15_controlnet")
    registry: dict[str, type[_BaseSketchPipeline]] = {
        "sd15_controlnet": SD15ControlNetPipeline,
        "sdxl_controlnet": SDXLControlNetPipeline,
        "t2i_adapter": T2IAdapterPipeline,
    }
    if backend not in registry:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Choose one of: {sorted(registry)}."
        )
    logger.info("Building pipeline for backend '%s'.", backend)
    return registry[backend](config)
