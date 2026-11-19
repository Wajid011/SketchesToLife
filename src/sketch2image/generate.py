"""High-level sketch-to-image generation orchestrator, style presets, and CLI.

Features:
- Prompt Helper: Library of style presets combining subject, style, materials,
  lighting, and camera terms, plus a robust default negative prompt.
- Generation Runner: Executes the unified :class:`~sketch2image.pipelines.SketchToImageGenerator`.
- JSON Sidecars: Every output image is paired with a comprehensive metadata JSON
  file documenting all hyper-parameters, model IDs, seed, runtime, and VRAM peak.
- Side-by-Side Visualisation: Generates comparison panels (Sketch | Condition | Output).
- Resumable: Skips generation if image and metadata already exist.
- CLI: ``python -m sketch2image.generate --sketch sketches/sketch_01.png --prompt "..."``
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from sketch2image.pipelines import SketchToImageGenerator
from sketch2image.preprocessing import prepare_condition
from sketch2image.utils import load_config, resolve_output_dir
from sketch2image.visualize import save_side_by_side

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style Presets Library
# ---------------------------------------------------------------------------

DEFAULT_NEGATIVE_PROMPT: str = (
    "blurry, low quality, bad anatomy, worst quality, watermark, text, signature, "
    "deformed, ugly, cropped, jpeg artifacts, lowres, distorted features, poorly drawn"
)

STYLE_PRESETS: dict[str, dict[str, str]] = {
    "photorealistic": {
        "suffix": (
            ", 8k UHD, high resolution photograph, sharp focus, natural daylight, "
            "professional photography, realistic texture, 35mm lens, f/1.8"
        ),
        "negative": ", cartoon, illustration, drawing, painting, 3d render, artificial",
    },
    "architectural_render": {
        "suffix": (
            ", modern architectural visualization, photorealistic architectural "
            "render, octane render, Unreal Engine 5, realistic materials, warm "
            "ambient evening lighting, ArchDaily award winner"
        ),
        "negative": ", sketch, drawing, low resolution, miniature, deformed geometry",
    },
    "concept_art": {
        "suffix": (
            ", digital concept art, trending on ArtStation, dynamic cinematic "
            "lighting, intricate details, atmospheric perspective, highly detailed "
            "digital illustration"
        ),
        "negative": ", photorealistic, photo, amateur, messy lines",
    },
    "product_studio": {
        "suffix": (
            ", clean commercial product studio photography, softbox studio "
            "lighting, soft neutral background, pristine reflections, 8k, sharp "
            "focus, commercial catalog shot"
        ),
        "negative": ", outdoor, busy background, harsh shadows, noisy, dirty",
    },
    "watercolor": {
        "suffix": (
            ", expressive watercolor painting, delicate brushstrokes, rough paper "
            "grain texture, soft color bleed, translucent washes, handcrafted "
            "artistic masterpiece"
        ),
        "negative": ", photograph, 3d render, sharp digital edges, plastic",
    },
    "cyberpunk": {
        "suffix": (
            ", cyberpunk aesthetic, vivid glowing neon lights, rainy wet asphalt "
            "reflections, moody volumetric haze, dark futuristic city atmosphere, "
            "high contrast, cinematic"
        ),
        "negative": ", sunny, daytime, pastoral, rustic, monochrome",
    },
    "anime": {
        "suffix": (
            ", modern Japanese anime style illustration, Makoto Shinkai aesthetic, "
            "clean crisp lines, vibrant sky with volumetric clouds, beautiful anime"
            " visual, masterpiece"
        ),
        "negative": ", photorealistic, 3d render, dark, gritty, Western comic",
    },
    "oil_painting": {
        "suffix": (
            ", classical oil painting on canvas, heavy impasto brushwork, rich "
            "textural depth, dramatic chiaroscuro lighting, Rembrandt palette, "
            "museum masterpiece"
        ),
        "negative": ", digital art, smooth, plastic, photograph, vector",
    },
}


def apply_style_preset(
    prompt: str,
    preset: str | None = None,
    custom_negative: str = "",
) -> tuple[str, str]:
    """Combine a prompt with style preset terms and default negative prompts.

    Args:
        prompt: Base user prompt describing the subject.
        preset: Name of style preset (e.g. 'photorealistic', 'cyberpunk').
        custom_negative: Additional negative prompt terms.

    Returns:
        Tuple of (enhanced_prompt, full_negative_prompt).
    """
    styled_prompt = prompt.strip()
    full_negative = DEFAULT_NEGATIVE_PROMPT

    if preset and preset.lower() in STYLE_PRESETS:
        info = STYLE_PRESETS[preset.lower()]
        styled_prompt += info["suffix"]
        full_negative += info["negative"]

    if custom_negative:
        full_negative = f"{full_negative}, {custom_negative.strip()}"

    return styled_prompt, full_negative


# ---------------------------------------------------------------------------
# Generation Runner
# ---------------------------------------------------------------------------


def generate_for_sketch(
    sketch_path: str | Path,
    prompt: str,
    generator: SketchToImageGenerator,
    output_dir: str | Path,
    preset: str | None = None,
    negative_prompt: str = "",
    condition_mode: str = "clean",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 1.0,
    resolution: int | None = None,
    seed: int = 42,
    num_images_per_prompt: int = 1,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Execute sketch-to-image generation and save images, sidecars, and comparison grids.

    Args:
        sketch_path: Path to sketch image.
        prompt: Base text prompt.
        generator: Configured :class:`SketchToImageGenerator`.
        output_dir: Directory where outputs are saved.
        preset: Optional style preset name.
        negative_prompt: Extra negative prompt.
        condition_mode: Conditioning map extractor ('clean', 'canny', etc.).
        num_inference_steps: Sampling steps.
        guidance_scale: Classifier-free guidance scale.
        controlnet_conditioning_scale: ControlNet strength.
        control_guidance_start: Step fraction start.
        control_guidance_end: Step fraction end.
        resolution: Output size.
        seed: RNG seed.
        num_images_per_prompt: Number of images to generate.
        skip_existing: If True, skip if outputs and JSON sidecar already exist.

    Returns:
        List of result dictionaries containing image paths and metadata.
    """
    sketch_path = Path(sketch_path)
    if not sketch_path.exists():
        raise FileNotFoundError(f"Sketch file not found: {sketch_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = sketch_path.stem
    styled_prompt, full_negative = apply_style_preset(
        prompt, preset=preset, custom_negative=negative_prompt
    )

    # File naming incorporating all key experiment parameters
    preset_tag = f"_{preset}" if preset else ""
    cn_tag = f"_{generator.controlnet_type}" if generator.controlnet_type else ""
    cond_tag = f"_{condition_mode}" if condition_mode != "clean" else ""
    scale_tag = (
        f"_scale{controlnet_conditioning_scale}"
        if controlnet_conditioning_scale != 1.0
        else ""
    )
    cfg_tag = f"_cfg{guidance_scale}" if guidance_scale != 7.5 else ""
    end_tag = f"_end{control_guidance_end}" if control_guidance_end != 1.0 else ""
    steps_tag = f"_steps{num_inference_steps}" if num_inference_steps != 30 else ""
    sched_clean = generator.scheduler_name.replace(" ", "").replace("+", "p")
    sched_tag = (
        f"_{sched_clean}"
        if sched_clean not in ("DPMpp2MKarras", "DPMSolverMultistep")
        else ""
    )

    base_name = (
        f"{stem}{preset_tag}_{generator.backend}{cn_tag}{cond_tag}"
        f"{scale_tag}{cfg_tag}{end_tag}{steps_tag}{sched_tag}_seed{seed}"
    )

    # Check for existing results (resumable)
    img_path = out_dir / f"{base_name}_0.png"
    meta_path = out_dir / f"{base_name}_meta.json"
    grid_path = out_dir / f"{base_name}_grid.png"

    if skip_existing and img_path.exists() and meta_path.exists():
        logger.info("Outputs already exist for '%s'; skipping.", base_name)
        with meta_path.open("r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        return [{"image_path": img_path, "metadata": cached_meta, "sidecar": meta_path}]

    logger.info(
        "Generating image for sketch '%s' with preset '%s'...", stem, preset or "none"
    )

    # 1. Generate condition map for preview / side-by-side
    target_res = resolution or (1024 if generator.backend == "sdxl_controlnet" else 512)
    cond_map = prepare_condition(
        sketch_path, mode=condition_mode, resolution=target_res
    )
    cond_file = out_dir / f"{base_name}_condition.png"
    cond_map.save(cond_file)

    # 2. Run generation
    images, metadata = generator.generate(
        sketch=sketch_path,
        prompt=styled_prompt,
        negative_prompt=full_negative,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        resolution=target_res,
        seed=seed,
        num_images_per_prompt=num_images_per_prompt,
        condition_mode=condition_mode,
    )

    # Add extra experiment metadata
    metadata.update(
        {
            "sketch_file": str(sketch_path.resolve()),
            "raw_prompt": prompt,
            "style_preset": preset,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "generated_files": [],
        }
    )

    results = []
    original_sketch = Image.open(sketch_path).convert("RGB")

    for idx, img in enumerate(images):
        single_img_file = out_dir / f"{base_name}_{idx}.png"
        img.save(single_img_file)
        metadata["generated_files"].append(str(single_img_file.name))

        # Save side-by-side comparison grid for first image
        if idx == 0:
            save_side_by_side(original_sketch, cond_map, img, grid_path)

        results.append(
            {
                "image_path": single_img_file,
                "metadata": metadata,
                "sidecar": meta_path,
                "grid_path": grid_path,
            }
        )

    # Save JSON sidecar
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata sidecar: %s", meta_path)

    return results


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for sketch2image.generate."""
    parser = argparse.ArgumentParser(
        description="Generate realistic images from hand-drawn sketches using diffusion."
    )
    parser.add_argument(
        "--sketch",
        "-s",
        required=True,
        help="Path to input sketch (PNG/JPG) or directory of sketches.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Description of the scene to generate.",
    )
    parser.add_argument(
        "--preset",
        choices=list(STYLE_PRESETS.keys()),
        default=None,
        help="Style preset (e.g. photorealistic, architectural_render, watercolor, cyberpunk).",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=["sd15_controlnet", "sdxl_controlnet", "t2i_adapter"],
        default=None,
        help="Override diffusion backend.",
    )
    parser.add_argument(
        "--controlnet-type",
        choices=["scribble", "lineart", "softedge"],
        default=None,
        help="ControlNet variant for SD 1.5.",
    )
    parser.add_argument(
        "--condition-mode",
        choices=["clean", "canny", "none", "hed", "lineart"],
        default=None,
        help="Sketch preprocessing condition extractor mode.",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of inference steps."
    )
    parser.add_argument(
        "--scale", type=float, default=None, help="ControlNet conditioning scale."
    )
    parser.add_argument(
        "--guidance", type=float, default=None, help="Classifier-free guidance scale."
    )
    parser.add_argument("--seed", type=int, default=None, help="Reproducibility seed.")
    parser.add_argument(
        "--num-images", type=int, default=1, help="Number of images to generate."
    )
    parser.add_argument("--out", "-o", default=None, help="Output directory.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run mock generation on CPU (for local development/testing without GPU).",
    )
    args = parser.parse_args()

    # Load base config
    cfg = load_config(args.config if Path(args.config).exists() else None)

    backend = args.backend or cfg.get("backend", "sd15_controlnet")
    cn_type = args.controlnet_type or cfg.get("controlnet_type", "scribble")
    cond_mode = args.condition_mode or cfg.get("preprocessor", "clean")
    steps = args.steps or cfg.get("num_inference_steps", 30)
    # `is not None`, not `or`: --scale 0 (ignore the sketch) is a valid setting.
    scale = (
        args.scale
        if args.scale is not None
        else cfg.get("controlnet_conditioning_scale", 1.0)
    )
    guidance = args.guidance or cfg.get("guidance_scale", 7.5)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    out_dir = Path(args.out) if args.out else resolve_output_dir(cfg)

    # Initialize generator
    generator = SketchToImageGenerator(
        backend=backend,
        controlnet_type=cn_type,
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "float16"),
        model_ids=cfg.get("models"),
        enable_attention_slicing=cfg.get("enable_attention_slicing", True),
        enable_model_cpu_offload=cfg.get("enable_model_cpu_offload", False),
        enable_vae_tiling=cfg.get("enable_vae_tiling", True),
        scheduler_name=cfg.get("scheduler", "DPM++ 2M Karras"),
        mock=args.mock,
    )

    sketch_target = Path(args.sketch)
    if sketch_target.is_file():
        sketch_files = [sketch_target]
    elif sketch_target.is_dir():
        sketch_files = sorted(
            [
                p
                for p in sketch_target.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
        )
    else:
        print(f"Error: Sketch path '{sketch_target}' not found.")
        return

    print("\n========================================================")
    print("  SketchesToLife — Generation")
    print("========================================================")
    print(f"  Backend:         {backend}")
    print(f"  ControlNet:      {cn_type}")
    print(f"  Condition Mode:  {cond_mode}")
    print(f"  Preset:          {args.preset or 'None'}")
    print(f"  Steps:           {steps}")
    print(f"  Scale:           {scale}")
    print(f"  Seed:            {seed}")
    print(f"  Output Dir:      {out_dir}")
    print(f"  Mock:            {args.mock}")
    print("========================================================\n")

    for sketch_file in sketch_files:
        print(f"Processing '{sketch_file.name}'...")
        results = generate_for_sketch(
            sketch_path=sketch_file,
            prompt=args.prompt,
            generator=generator,
            output_dir=out_dir,
            preset=args.preset,
            condition_mode=cond_mode,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=scale,
            seed=seed,
            num_images_per_prompt=args.num_images,
        )
        for res in results:
            print(f"  ✓ Saved image:   {res['image_path']}")
            print(f"  ✓ Saved sidecar: {res['sidecar']}")
            if "grid_path" in res:
                print(f"  ★ Saved grid:    {res['grid_path']}")
        print()


if __name__ == "__main__":
    main()
