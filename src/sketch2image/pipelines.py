"""Diffusion pipeline wrappers and backend registry for sketch-to-image generation.

Provides the :class:`SketchToImageGenerator` class managing:
- Multiple backends:
  * ``sd15_controlnet``: StableDiffusionControlNetPipeline with ControlNet v1.1
    (scribble, lineart, softedge checkpoints).
  * ``sdxl_controlnet``: StableDiffusionXLControlNetPipeline with SDXL ControlNet
    scribble + optional fp16 VAE fix (madebyollin/sdxl-vae-fp16-fix).
  * ``t2i_adapter``: StableDiffusionAdapterPipeline with TencentARC T2I-Adapter sketch.
- Lazy model loading, pipeline caching, and GPU memory release.
- fp16 precision, attention slicing, CPU offloading, and VAE tiling for low-VRAM (T4 / 16 GB).
- Schedulers: DPM++ 2M Karras, Euler a, UniPC, DDIM.
- Reproducible multi-image generation with seeded torch.Generator.
- Structured execution metadata (generation time, peak VRAM, parameters).
- Mock/simulation mode for local CPU development without downloading multi-GB weights.
"""

from __future__ import annotations

import gc
import logging
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sketch2image.preprocessing import prepare_condition
from sketch2image.utils import set_seed

logger = logging.getLogger(__name__)

# Verified Hugging Face Model IDs
DEFAULT_MODEL_IDS: dict[str, str] = {
    # SD 1.5
    "sd15_base": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "controlnet_scribble": "lllyasviel/control_v11p_sd15_scribble",
    "controlnet_lineart": "lllyasviel/control_v11p_sd15_lineart",
    "controlnet_softedge": "lllyasviel/control_v11p_sd15_softedge",
    # SDXL
    "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0",
    "controlnet_sdxl_scribble": "xinsir/controlnet-scribble-sdxl-1.0",
    "sdxl_vae_fix": "madebyollin/sdxl-vae-fp16-fix",
    # T2I-Adapter
    "t2i_adapter_sketch": "TencentARC/t2iadapter_sketch_sd15v2",
}


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------


def _enable_vae_tiling(pipeline: Any) -> None:
    """Decode the latent in tiles to save VRAM.

    Called on the VAE itself: ``pipeline.enable_vae_tiling()`` was removed from
    recent diffusers releases, while ``vae.enable_tiling()`` exists in all of them.
    """
    vae = getattr(pipeline, "vae", None)
    if vae is not None and callable(getattr(vae, "enable_tiling", None)):
        vae.enable_tiling()


def _match_adapter_channels(pipeline: Any, image: Image.Image) -> Image.Image:
    """Give a T2I-Adapter the number of image channels it was trained on.

    The sketch adapter (``in_channels: 1``) expects a grayscale map; an RGB map
    would reach its first convolution with 3x too many channels.
    """
    adapter = getattr(pipeline, "adapter", None)
    in_channels = getattr(getattr(adapter, "config", None), "in_channels", None)
    if in_channels == 1:
        return image.convert("L")
    return image


def configure_scheduler(pipeline: Any, scheduler_name: str) -> None:
    """Configure or replace the pipeline's noise scheduler.

    Supports:
    - ``'DPM++ 2M Karras'`` / ``'DPMSolverMultistep'``
    - ``'Euler a'`` / ``'EulerAncestral'``
    - ``'UniPC'`` / ``'UniPCMultistep'``
    - ``'DDIM'``
    - ``'PNDM'``

    Args:
        pipeline: Instantiated diffusers pipeline.
        scheduler_name: Name of the scheduler.
    """
    try:
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            PNDMScheduler,
            UniPCMultistepScheduler,
        )
    except ImportError:
        logger.debug("diffusers not installed; scheduler configuration skipped.")
        return

    name_lower = scheduler_name.lower().replace("-", " ").replace("_", " ")

    if "dpm" in name_lower:
        use_karras = "karras" in name_lower or True
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config, use_karras_sigmas=use_karras
        )
    elif "euler a" in name_lower or "ancestral" in name_lower:
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )
    elif "euler" in name_lower:
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )
    elif "unipc" in name_lower:
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
    elif "ddim" in name_lower:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif "pndm" in name_lower:
        pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)
    else:
        logger.warning("Unrecognized scheduler '%s'; leaving existing.", scheduler_name)


# ---------------------------------------------------------------------------
# Unified Generator Class
# ---------------------------------------------------------------------------


class SketchToImageGenerator:
    """Unified generator supporting SD 1.5 ControlNet, SDXL ControlNet, and T2I-Adapter.

    Features:
    - Lazy loading & caching: loads pipeline on first demand, caches for reuse.
    - Low-VRAM optimizations for Colab T4: fp16, attention slicing, CPU offload, VAE tiling.
    - CPU/Mock fallback: safe local development and testing without GPU/model downloads.
    """

    def __init__(
        self,
        backend: str = "sd15_controlnet",
        controlnet_type: str = "scribble",
        device: str = "cuda",
        dtype: str = "float16",
        model_ids: dict[str, str] | None = None,
        enable_attention_slicing: bool = True,
        enable_model_cpu_offload: bool = False,
        enable_vae_tiling: bool = True,
        scheduler_name: str = "DPM++ 2M Karras",
        mock: bool = False,
    ) -> None:
        self.backend = backend.lower().strip()
        self.controlnet_type = controlnet_type.lower().strip()
        self.device = device
        self.dtype_str = dtype
        self.model_ids = {**DEFAULT_MODEL_IDS, **(model_ids or {})}
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_vae_tiling = enable_vae_tiling
        self.scheduler_name = scheduler_name
        self.mock = mock

        # Cached active pipeline
        self._pipe: Any = None
        self._loaded_cache_key: str | None = None

    # ------------------------------------------------------------------
    # Pipeline Loading & Caching
    # ------------------------------------------------------------------

    def _get_cache_key(self) -> str:
        return f"{self.backend}:{self.controlnet_type}:{self.device}:{self.dtype_str}"

    def load(self) -> None:
        """Load the configured pipeline into memory (cached if already loaded)."""
        cache_key = self._get_cache_key()
        if self._pipe is not None and self._loaded_cache_key == cache_key:
            logger.debug("Pipeline already loaded with key: %s", cache_key)
            return

        valid_backends = ("sd15_controlnet", "sdxl_controlnet", "t2i_adapter")
        if self.backend not in valid_backends:
            raise ValueError(
                f"Unknown backend: '{self.backend}'. "
                f"Choose from: {sorted(valid_backends)}."
            )

        if self.mock or not self._is_torch_cuda_available():
            logger.info("Operating in mock/simulation mode (no GPU/torch available).")
            self._pipe = "mock_pipeline"
            self._loaded_cache_key = cache_key
            return

        logger.info(
            "Loading pipeline: backend='%s', controlnet='%s'...",
            self.backend,
            self.controlnet_type,
        )
        import torch

        torch_dtype = torch.float16 if self.dtype_str == "float16" else torch.float32

        if self.backend == "sd15_controlnet":
            self._load_sd15_controlnet(torch_dtype)
        elif self.backend == "sdxl_controlnet":
            self._load_sdxl_controlnet(torch_dtype)
        elif self.backend == "t2i_adapter":
            self._load_t2i_adapter(torch_dtype)

        configure_scheduler(self._pipe, self.scheduler_name)
        self._loaded_cache_key = cache_key
        logger.info("Pipeline ready.")

    def _load_sd15_controlnet(self, torch_dtype: Any) -> None:
        """Load SD 1.5 + ControlNet v1.1."""
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

        cn_key = f"controlnet_{self.controlnet_type}"
        cn_id = self.model_ids.get(cn_key, self.model_ids["controlnet_scribble"])
        base_id = self.model_ids["sd15_base"]

        logger.info("Loading ControlNet: %s", cn_id)
        controlnet = ControlNetModel.from_pretrained(cn_id, torch_dtype=torch_dtype)

        logger.info("Loading base SD 1.5: %s", base_id)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )

        if self.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        if self.enable_attention_slicing:
            pipe.enable_attention_slicing()
        if self.enable_vae_tiling:
            _enable_vae_tiling(pipe)

        self._pipe = pipe

    def _load_sdxl_controlnet(self, torch_dtype: Any) -> None:
        """Load SDXL + ControlNet with CPU offload and fp16 VAE fix."""
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
        )

        cn_id = self.model_ids["controlnet_sdxl_scribble"]
        base_id = self.model_ids["sdxl_base"]
        vae_fix_id = self.model_ids.get("sdxl_vae_fix")

        logger.info("Loading SDXL ControlNet: %s", cn_id)
        controlnet = ControlNetModel.from_pretrained(
            cn_id, torch_dtype=torch_dtype, use_safetensors=True
        )

        vae = None
        if vae_fix_id:
            try:
                logger.info("Loading fixed SDXL VAE: %s", vae_fix_id)
                vae = AutoencoderKL.from_pretrained(vae_fix_id, torch_dtype=torch_dtype)
            except Exception as exc:
                logger.warning("Could not load fixed VAE (%s); using default.", exc)

        pipe_kwargs: dict[str, Any] = {
            "controlnet": controlnet,
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
        }
        if vae is not None:
            pipe_kwargs["vae"] = vae

        logger.info("Loading SDXL base: %s", base_id)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_id, **pipe_kwargs
        )

        # Mandatory CPU offload for SDXL on 16GB T4
        pipe.enable_model_cpu_offload()

        if self.enable_attention_slicing:
            pipe.enable_attention_slicing()
        if self.enable_vae_tiling:
            _enable_vae_tiling(pipe)

        self._pipe = pipe

    def _load_t2i_adapter(self, torch_dtype: Any) -> None:
        """Load SD 1.5 + T2I-Adapter Sketch."""
        from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

        adapter_id = self.model_ids["t2i_adapter_sketch"]
        base_id = self.model_ids["sd15_base"]

        logger.info("Loading T2I-Adapter: %s", adapter_id)
        adapter = T2IAdapter.from_pretrained(adapter_id, torch_dtype=torch_dtype)

        logger.info("Loading base SD 1.5: %s", base_id)
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            base_id,
            adapter=adapter,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )

        if self.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        if self.enable_attention_slicing:
            pipe.enable_attention_slicing()

        self._pipe = pipe

    def unload(self) -> None:
        """Unload pipeline and release GPU memory."""
        self._pipe = None
        self._loaded_cache_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Pipeline unloaded.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        sketch: str | Path | Image.Image | np.ndarray,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        resolution: int | None = None,
        seed: int = 42,
        num_images_per_prompt: int = 1,
        condition_mode: str = "clean",
        condition_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        """Generate images conditioned on sketch.

        Args:
            sketch: Input sketch image or file path.
            prompt: Text prompt describing the target scene.
            negative_prompt: Text prompt describing unwanted artifacts.
            num_inference_steps: Number of diffusion sampling steps.
            guidance_scale: Classifier-free guidance strength.
            controlnet_conditioning_scale: ControlNet conditioning strength.
            control_guidance_start: Fraction of steps where ControlNet activates.
            control_guidance_end: Fraction of steps where ControlNet turns off.
            resolution: Output image size (default 512 for SD 1.5, 1024 for SDXL).
            seed: Integer random seed for reproducibility.
            num_images_per_prompt: Number of images to generate.
            condition_mode: Preprocessing mode ('clean', 'canny', 'none', etc.).
            condition_kwargs: Extra options for
                :func:`~sketch2image.preprocessing.prepare_condition`
                (e.g. ``{"line_thickness": 1}``).
            **kwargs: Extra arguments passed to the pipeline.

        Returns:
            Tuple of (list_of_generated_PIL_images, metadata_dict).
        """
        # Determine resolution default based on backend
        if resolution is None:
            resolution = 1024 if self.backend == "sdxl_controlnet" else 512

        # 1. Preprocess the sketch into canonical conditioning map
        conditioning_map = prepare_condition(
            sketch,
            mode=condition_mode,
            resolution=resolution,
            multiple=64,
            **(condition_kwargs or {}),
        )

        # 2. Ensure pipeline is loaded
        self.load()

        # 3. Track execution time and peak VRAM
        self._reset_peak_vram()
        start_time = time.perf_counter()

        # 4. Run diffusion (or mock if no GPU/torch)
        if self._pipe == "mock_pipeline":
            images = self._mock_generate(
                conditioning_map, prompt, resolution, num_images_per_prompt, seed
            )
        else:
            images = self._run_diffusion(
                conditioning_map=conditioning_map,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                resolution=resolution,
                seed=seed,
                num_images_per_prompt=num_images_per_prompt,
                **kwargs,
            )

        self._cuda_synchronize()
        runtime_sec = round(time.perf_counter() - start_time, 2)
        # Absolute peak (weights + activations) for this call, not a delta:
        # max_memory_allocated() is a process-wide high-water mark, so it is
        # reset above and read here.
        vram_peak_mb = self._get_vram_mb()

        # 5. Compile metadata sidecar
        metadata: dict[str, Any] = {
            "backend": self.backend,
            "controlnet_type": self.controlnet_type,
            "condition_mode": condition_mode,
            "model_ids": self.model_ids,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "resolution": resolution,
            "seed": seed,
            "scheduler": self.scheduler_name,
            "num_images": len(images),
            "runtime_sec": runtime_sec,
            "peak_vram_mb": max(0.0, round(vram_peak_mb, 1)),
            "mock": self._pipe == "mock_pipeline",
        }

        return images, metadata

    def _run_diffusion(
        self,
        conditioning_map: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
        control_guidance_start: float,
        control_guidance_end: float,
        resolution: int,
        seed: int,
        num_images_per_prompt: int,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Execute real diffusers pipeline on GPU."""
        import torch

        # Seeds python/numpy/torch and enables cuDNN deterministic mode; the
        # explicit torch.Generator below is what fixes the initial latent noise.
        set_seed(seed)
        generator = torch.Generator(
            device="cpu" if self.enable_model_cpu_offload else self.device
        )
        generator.manual_seed(seed)

        call_kwargs: dict[str, Any] = {
            "prompt": [prompt] * num_images_per_prompt,
            "image": _match_adapter_channels(self._pipe, conditioning_map),
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": resolution,
            "width": resolution,
            "generator": generator,
            **kwargs,
        }
        if negative_prompt:
            call_kwargs["negative_prompt"] = [negative_prompt] * num_images_per_prompt

        if self.backend in ("sd15_controlnet", "sdxl_controlnet"):
            call_kwargs["controlnet_conditioning_scale"] = float(
                controlnet_conditioning_scale
            )
            call_kwargs["control_guidance_start"] = float(control_guidance_start)
            call_kwargs["control_guidance_end"] = float(control_guidance_end)

        result = self._pipe(**call_kwargs)
        return result.images

    # ------------------------------------------------------------------
    # Mock / Simulation Generator (for local CPU test & review)
    # ------------------------------------------------------------------

    def _mock_generate(
        self,
        conditioning_map: Image.Image,
        prompt: str,
        resolution: int,
        num_images: int,
        seed: int,
    ) -> list[Image.Image]:
        """Generate stylized mock images for testing on CPU when diffusion models are absent."""
        import cv2

        cond_arr = np.array(conditioning_map.resize((resolution, resolution)))
        images = []

        for i in range(num_images):
            # Create a stylized mock image incorporating prompt colors and sketch edges
            # zlib.crc32, not hash(): str hashes are salted per process
            # (PYTHONHASHSEED), which made mock outputs differ between runs.
            color_hue = zlib.crc32(f"{prompt}{seed}{i}".encode()) % 180
            base_hsv = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            base_hsv[..., 0] = color_hue
            base_hsv[..., 1] = 120
            base_hsv[..., 2] = 200

            # Add gradient lighting
            y, x = np.mgrid[0:resolution, 0:resolution]
            gradient = (((x + y) / (2 * resolution)) * 80).astype(np.uint8)
            base_hsv[..., 2] = np.clip(base_hsv[..., 2] - gradient, 40, 255)

            base_bgr = cv2.cvtColor(base_hsv, cv2.COLOR_HSV2BGR)

            # Composite the sketch lines as vivid foreground / lighting
            edge_mask = (cond_arr[..., 0] > 50).astype(np.float32)[..., None]
            line_color = np.array([255, 255, 255], dtype=np.float32)
            composite = base_bgr * (1.0 - edge_mask * 0.7) + line_color * (
                edge_mask * 0.7
            )
            composite = np.clip(composite, 0, 255).astype(np.uint8)

            # Convert BGR to RGB
            rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))

        return images

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _is_torch_cuda_available(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _reset_peak_vram(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def _cuda_synchronize(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    def _get_vram_mb(self) -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        return 0.0


# Backward compatibility factory
def build_pipeline(config: dict[str, Any]) -> Any:
    """Create a SketchToImageGenerator from config dictionary."""
    return SketchToImageGenerator(
        backend=config.get("backend", "sd15_controlnet"),
        controlnet_type=config.get("controlnet_type", "scribble"),
        device=config.get("device", "cuda"),
        dtype=config.get("dtype", "float16"),
        model_ids=config.get("models"),
        enable_attention_slicing=config.get("enable_attention_slicing", True),
        enable_model_cpu_offload=config.get("enable_model_cpu_offload", False),
        enable_vae_tiling=config.get("enable_vae_tiling", True),
        scheduler_name=config.get("scheduler", "DPM++ 2M Karras"),
    )
