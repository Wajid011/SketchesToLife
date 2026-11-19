#!/usr/bin/env python3
"""scripts/run_experiments.py — Systematic experiment runner for SketchesToLife.

Runs sweeps defined in ``configs/experiments/*.yaml``:
- E1: Backend & ControlNet Comparison (SD1.5 Scribble/Lineart/SoftEdge vs SDXL vs T2I-Adapter)
- E2: Conditioning Scale (0.2 -> 1.5)
- E3: Guidance Scale (3.0 -> 13.0)
- E4: Control Guidance Release in Late Steps (0.4 -> 1.0)
- E5: Preprocessing Conditioning Map Comparison (Raw vs Clean vs Canny vs Lineart)
- E6: Style Presets & Artistic Variation
  (Photorealistic, Arch Render, Concept Art, Watercolor, Cyberpunk)
- E7: Schedulers & Steps Trade-off
- E8: Random Seed Variability (Stochastic consistency)

Features:
- Fixed seeds for rigorous scientific reproducibility.
- Resumable: automatically skips image generation if output image & metadata JSON exist.
- Generates publication-ready comparison figures in ``results/`` with input sketch on the left.
- Generates a "hero" figure showcasing the premier output for each sketch.
- Auto-detects environment: uses CUDA on GPU (Colab/Kaggle) or mock CPU generation locally.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure src/ is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from sketch2image.generate import generate_for_sketch  # noqa: E402
from sketch2image.pipelines import SketchToImageGenerator  # noqa: E402
from sketch2image.visualize import (  # noqa: E402
    create_2d_matrix_grid,
    create_experiment_grid,
    create_hero_figure,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Standard academic prompts tailored to each sample sketch
SKETCH_PROMPTS: dict[str, str] = {
    "sketch_01": "cozy wooden cabin in the rocky mountains, warm glowing windows, pine forest",
    "sketch_02": "futuristic sleek electric sports car, reflections on glossy metallic red paint",
    "sketch_03": "cute distinguished gentleman cat",
}


def load_experiment_config(yaml_path: Path) -> dict[str, Any]:
    """Load and parse an experiment configuration file."""
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single_experiment(
    exp_cfg: dict[str, Any],
    sketch_path: Path,
    out_base_dir: Path,
    results_dir: Path,
    mock: bool = False,
) -> tuple[list[Path], list[str], Path]:
    """Execute a single experiment sweep on one sketch and save its publication grid.

    Returns:
        Tuple of (list_of_variant_image_paths, list_of_labels, saved_grid_path).
    """
    exp_id = exp_cfg["id"]
    exp_title = exp_cfg.get("title", exp_id)
    stem = sketch_path.stem
    prompt = SKETCH_PROMPTS.get(stem, "a high quality detailed artwork")

    exp_out_dir = out_base_dir / exp_id / stem
    exp_out_dir.mkdir(parents=True, exist_ok=True)

    fixed_params = exp_cfg.get("fixed", {})
    sweep = exp_cfg.get("sweep", {})
    sweep_type = sweep.get("type", "scalar_sweep")

    logger.info("Running %s on '%s' (sweep type: %s)...", exp_id, stem, sweep_type)

    variant_images: list[Path] = []
    variant_labels: list[str] = []

    # ------------------------------------------------------------------
    # Case A: Variant List (e.g. E1: different backends & models)
    # ------------------------------------------------------------------
    if sweep_type == "variant_list":
        for var in sweep["variants"]:
            label = var["label"]
            merged = {**fixed_params, **var}
            backend = merged["backend"]
            cn_type = merged.get("controlnet_type", "scribble")
            res = merged.get("resolution", 512)

            generator = SketchToImageGenerator(
                backend=backend,
                controlnet_type=cn_type,
                mock=mock,
            )

            results = generate_for_sketch(
                sketch_path=sketch_path,
                prompt=prompt,
                generator=generator,
                output_dir=exp_out_dir,
                condition_mode=merged.get("condition_mode", "clean"),
                num_inference_steps=merged.get("num_inference_steps", 30),
                guidance_scale=merged.get("guidance_scale", 7.5),
                controlnet_conditioning_scale=merged.get(
                    "controlnet_conditioning_scale", 1.0
                ),
                resolution=res,
                seed=merged.get("seed", 42),
                num_images_per_prompt=1,
            )
            variant_images.append(results[0]["image_path"])
            variant_labels.append(label)

    # ------------------------------------------------------------------
    # Case B: Scalar Sweep (e.g. E2 scale, E3 guidance, E4 release, E5 prep, E6 style, E8 seed)
    # ------------------------------------------------------------------
    elif sweep_type == "scalar_sweep":
        param_name = sweep["parameter"]
        values = sweep["values"]
        labels = sweep.get("labels", [f"{param_name}={v}" for v in values])

        backend = fixed_params.get("backend", "sd15_controlnet")
        cn_type = fixed_params.get("controlnet_type", "scribble")
        generator = SketchToImageGenerator(
            backend=backend, controlnet_type=cn_type, mock=mock
        )

        for val, label in zip(values, labels, strict=False):
            curr_params = {**fixed_params, param_name: val}

            # Handle preset override
            preset_val = curr_params.get("preset") if param_name == "preset" else None

            results = generate_for_sketch(
                sketch_path=sketch_path,
                prompt=prompt,
                generator=generator,
                output_dir=exp_out_dir,
                preset=preset_val,
                condition_mode=curr_params.get("condition_mode", "clean"),
                num_inference_steps=curr_params.get("num_inference_steps", 30),
                guidance_scale=curr_params.get("guidance_scale", 7.5),
                controlnet_conditioning_scale=curr_params.get(
                    "controlnet_conditioning_scale", 1.0
                ),
                control_guidance_start=curr_params.get("control_guidance_start", 0.0),
                control_guidance_end=curr_params.get("control_guidance_end", 1.0),
                resolution=curr_params.get("resolution", 512),
                seed=curr_params.get("seed", 42),
                num_images_per_prompt=1,
            )
            variant_images.append(results[0]["image_path"])
            variant_labels.append(label)

    # ------------------------------------------------------------------
    # Case C: 2D Grid (e.g. E7: steps x schedulers)
    # ------------------------------------------------------------------
    elif sweep_type == "2d_grid":
        r_param = sweep["row_parameter"]
        r_vals = sweep["row_values"]
        r_labels = sweep.get("row_labels", [str(v) for v in r_vals])

        c_param = sweep["col_parameter"]
        c_vals = sweep["col_values"]
        c_labels = sweep.get("col_labels", [str(v) for v in c_vals])

        matrix_images: list[list[Path]] = []
        for r_val in r_vals:
            row_imgs: list[Path] = []
            for c_val in c_vals:
                curr_params = {**fixed_params, r_param: r_val, c_param: c_val}
                sched = curr_params.get("scheduler", "DPM++ 2M Karras")
                generator = SketchToImageGenerator(
                    backend=curr_params.get("backend", "sd15_controlnet"),
                    controlnet_type=curr_params.get("controlnet_type", "scribble"),
                    scheduler_name=sched,
                    mock=mock,
                )
                results = generate_for_sketch(
                    sketch_path=sketch_path,
                    prompt=prompt,
                    generator=generator,
                    output_dir=exp_out_dir,
                    num_inference_steps=curr_params.get("num_inference_steps", 30),
                    guidance_scale=curr_params.get("guidance_scale", 7.5),
                    controlnet_conditioning_scale=curr_params.get(
                        "controlnet_conditioning_scale", 1.0
                    ),
                    seed=curr_params.get("seed", 42),
                    num_images_per_prompt=1,
                )
                row_imgs.append(results[0]["image_path"])
            matrix_images.append(row_imgs)

        # Generate 2D Matrix Grid
        grid_out = results_dir / f"{exp_id}_{stem}_matrix_grid.png"
        create_2d_matrix_grid(
            sketch=sketch_path,
            matrix_images=matrix_images,
            row_labels=r_labels,
            col_labels=c_labels,
            output_path=grid_out,
            title=f"{exp_id}: {exp_title} — {stem.replace('_', ' ').title()}",
            subtitle=f'Prompt: "{prompt}"',
        )
        return [], [], grid_out

    # Generate 1D Sweep Publication Grid
    grid_out = results_dir / f"{exp_id}_{stem}_grid.png"
    subtitle_text = f"Prompt: \"{prompt}\" | Fixed Seed: {fixed_params.get('seed', 42)}"
    create_experiment_grid(
        sketch=sketch_path,
        generated_images=variant_images,
        column_labels=variant_labels,
        output_path=grid_out,
        title=f"{exp_id}: {exp_title} — {stem.replace('_', ' ').title()}",
        subtitle=subtitle_text,
    )

    return variant_images, variant_labels, grid_out


# Hero figure = one E6 (style preset) output per sketch, at E6's fixed seed.
HERO_CONFIGS = [
    {
        "stem": "sketch_01",
        "caption": "Cabin — Photorealistic SD1.5",
        "preset": "photorealistic",
    },
    {
        "stem": "sketch_02",
        "caption": "Sports Car — Concept Art SD1.5",
        "preset": "concept_art",
    },
    {"stem": "sketch_03", "caption": "Cat — Watercolor SD1.5", "preset": "watercolor"},
]


def run_hero_figure(sketches: list[Path], out_dir: Path, results_dir: Path) -> Path:
    """Build the hero figure from the exact E6 outputs named in ``HERO_CONFIGS``.

    Requires E6 to have been run into ``out_dir``. A missing image is an error:
    silently substituting another output would make the figure irreproducible.
    """
    hero_pairs = []
    for cfg in HERO_CONFIGS:
        stem = cfg["stem"]
        sk_path = next((p for p in sketches if p.stem == stem), None)
        if sk_path is None:
            continue
        # Name produced by generate_for_sketch for E6's fixed params
        # (sd15_controlnet + scribble, seed 42, all other settings at defaults).
        output = (
            out_dir
            / "E6"
            / stem
            / f"{stem}_{cfg['preset']}_sd15_controlnet_scribble_seed42_0.png"
        )
        if not output.exists():
            raise FileNotFoundError(
                f"Hero image missing: {output}. Run experiment E6 first (-e E6)."
            )
        hero_pairs.append(
            {"sketch": sk_path, "output": output, "caption": cfg["caption"]}
        )

    hero_out = results_dir / "hero_comparison.png"
    create_hero_figure(hero_pairs, output_path=hero_out)
    return hero_out


def main() -> None:
    """CLI entry point: run the selected sweeps, their grids and the hero figure."""
    parser = argparse.ArgumentParser(
        description="Run systematic sketch-to-image experiment sweeps."
    )
    parser.add_argument(
        "--experiments",
        "-e",
        nargs="+",
        default=["E1", "E2", "E6"],
        help="Experiment IDs to run (e.g. E1 E2 E6, or 'all'). Default: E1 E2 E6.",
    )
    parser.add_argument(
        "--sketches-dir",
        default="sketches",
        help="Directory containing hand-drawn sketch files.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/experiments",
        help="Directory containing experiment YAML files.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/experiments",
        help="Directory to save raw generated images and metadata.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save publication-quality grids and hero figures.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock/simulation mode (for testing on CPU without GPU/models).",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    sketches_dir = Path(args.sketches_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find available experiment configs
    exp_files = sorted(config_dir.glob("*.yaml"))
    if not exp_files:
        print(f"No experiment YAML files found in '{config_dir}'.")
        return

    exp_map = {f.stem.split("_")[0].upper(): f for f in exp_files}

    # Determine which experiments to run
    target_exp_ids = args.experiments
    if "all" in [x.lower() for x in target_exp_ids]:
        selected_files = exp_files
    else:
        selected_files = []
        for x in target_exp_ids:
            key = x.upper()
            if key in exp_map:
                selected_files.append(exp_map[key])
            else:
                print(
                    f"Warning: Unknown experiment ID '{x}'. Available: {list(exp_map.keys())}"
                )

    # Collect sketches
    sketch_files = sorted(
        [
            p
            for p in sketches_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
    )
    if not sketch_files:
        print(f"No sketches found in '{sketches_dir}'.")
        return

    print("=================================================================")
    print("  SketchesToLife — Experimentation Framework")
    print("=================================================================")
    print(
        f"  Target Experiments: {[f.stem.split('_')[0].upper() for f in selected_files]}"
    )
    print(f"  Sketches Found:     {[s.name for s in sketch_files]}")
    print(f"  Output Dir:         {out_dir}")
    print(f"  Results Dir:        {results_dir}")
    print(f"  Mock Mode:          {args.mock}")
    print("=================================================================\n")

    completed_grids: list[Path] = []

    for exp_file in selected_files:
        exp_cfg = load_experiment_config(exp_file)
        exp_id = exp_cfg["id"]
        print(f"▶▶ Running Experiment {exp_id}: {exp_cfg.get('title')}")
        print(f"   {exp_cfg.get('description', '').strip()}\n")

        for sketch_path in sketch_files:
            _, _, grid_path = run_single_experiment(
                exp_cfg=exp_cfg,
                sketch_path=sketch_path,
                out_base_dir=out_dir,
                results_dir=results_dir,
                mock=args.mock,
            )
            completed_grids.append(grid_path)
            print(f"   ✓ Generated grid: {grid_path}")
        print()

    # Generate the Hero figure
    print("★ Generating Flagship Hero Figure (All Sketches -> Best Output)...")
    if (out_dir / "E6").is_dir():
        hero_figure = run_hero_figure(sketch_files, out_dir, results_dir)
        print(f"   ✓ Saved hero figure: {hero_figure}\n")
    else:
        print("   – Skipped: the hero figure is built from E6 outputs (add -e E6).\n")

    print("=================================================================")
    print(f"  Experiment Sweep Complete! Generated {len(completed_grids)} grids.")
    print("=================================================================")


if __name__ == "__main__":
    main()
