"""High-level generation orchestrator.

``generate.py`` is the bridge between the CLI / Gradio UI and the
underlying pipeline classes. It:

1. Loads and merges YAML config with any runtime overrides.
2. Instantiates the correct preprocessor and pipeline.
3. Iterates over sketch paths, skipping existing outputs (resumable).
4. Saves the raw output, the conditioning map, and a side-by-side grid.

This module is intentionally thin — all heavy logic lives in
:mod:`sketch2image.pipelines` and :mod:`sketch2image.preprocessing`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from sketch2image.preprocessing import SketchPreprocessor, load_sketch
from sketch2image.utils import load_config, resolve_output_dir, set_seed
from sketch2image.visualize import save_side_by_side

logger = logging.getLogger(__name__)


def generate_single(
    sketch_path: str | Path,
    config: dict[str, Any],
    output_dir: Path,
    pipeline: Any,
    preprocessor: SketchPreprocessor,
) -> dict[str, Path]:
    """Generate one image from one sketch.

    Args:
        sketch_path: Path to the input sketch file.
        config: Merged experiment configuration.
        output_dir: Directory in which to save outputs.
        pipeline: Loaded pipeline instance (already called ``.load()``).
        preprocessor: Instantiated :class:`~sketch2image.preprocessing.SketchPreprocessor`.

    Returns:
        Dict with keys ``generated``, ``conditioning_map``, ``side_by_side``
        mapping to the saved file paths.
    """
    sketch_path = Path(sketch_path)
    stem = sketch_path.stem
    seed = config.get("seed", 42)
    scale = config.get("controlnet_conditioning_scale", 1.0)

    # Build output file names that encode the key params for easy sorting
    suffix = f"seed{seed}_scale{scale}"
    out_generated = output_dir / f"{stem}_{suffix}_generated.png"
    out_cond = output_dir / f"{stem}_{suffix}_cond_map.png"
    out_grid = output_dir / f"{stem}_{suffix}_grid.png"

    # Resumability: skip if all outputs already exist
    if out_generated.exists() and out_cond.exists() and out_grid.exists():
        logger.info("All outputs exist for '%s'; skipping.", sketch_path.name)
        return {
            "generated": out_generated,
            "conditioning_map": out_cond,
            "side_by_side": out_grid,
        }

    logger.info("Processing sketch: %s", sketch_path.name)

    # 1. Load and preprocess the sketch
    sketch = load_sketch(sketch_path, resolution=config.get("resolution", 512))
    cond_map = preprocessor(sketch)

    # 2. Set global seed for reproducibility
    set_seed(seed)

    # 3. Run inference
    generated = pipeline.generate(
        conditioning_map=cond_map,
        prompt=config["prompt"],
        negative_prompt=config.get("negative_prompt", ""),
        seed=seed,
    )

    # 4. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.get("save_conditioning_map", True):
        cond_map.save(out_cond)
        logger.debug("Saved conditioning map: %s", out_cond)

    generated.save(out_generated)
    logger.info("Saved generated image: %s", out_generated)

    if config.get("save_side_by_side", True):
        save_side_by_side(sketch, cond_map, generated, out_grid)
        logger.debug("Saved side-by-side grid: %s", out_grid)

    return {
        "generated": out_generated,
        "conditioning_map": out_cond,
        "side_by_side": out_grid,
    }


def run_generation(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Path]]:
    """Full generation run from a YAML config file.

    Loads config, builds pipeline + preprocessor, iterates over all
    ``sketch_paths`` in config, and returns a list of output path dicts.

    Args:
        config_path: Path to the YAML experiment config file.
        overrides: Optional dict of config keys to override at runtime.

    Returns:
        List of dicts (one per sketch) with keys ``generated``,
        ``conditioning_map``, ``side_by_side``.
    """
    # Import here so the module is importable without diffusers on CPU
    from sketch2image.pipelines import build_pipeline

    config = load_config(config_path, overrides=overrides)
    output_dir = resolve_output_dir(config)

    preprocessor = SketchPreprocessor(
        preprocessor_type=config.get("preprocessor", "hed"),
        resolution=config.get("preprocessor_resolution", 512),
    )

    pipeline = build_pipeline(config)
    pipeline.load()

    results = []
    sketch_paths = config.get("sketch_paths", [])
    logger.info("Running generation for %d sketch(es).", len(sketch_paths))

    for sketch_path in sketch_paths:
        try:
            result = generate_single(
                sketch_path=sketch_path,
                config=config,
                output_dir=output_dir,
                pipeline=pipeline,
                preprocessor=preprocessor,
            )
            results.append(result)
        except Exception:
            logger.exception("Failed to process '%s'; continuing.", sketch_path)

    pipeline.unload()
    logger.info("Generation complete. %d image(s) produced.", len(results))
    return results


# ---------------------------------------------------------------------------
# CLI entry point (wired in pyproject.toml as s2i-generate)
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the ``s2i-generate`` command."""
    import click

    @click.command()
    @click.option("--config", "-c", required=True, help="Path to experiment YAML config.")
    @click.option(
        "--override",
        "-o",
        multiple=True,
        help="key=value overrides, e.g. -o seed=123 -o guidance_scale=9",
    )
    def _cli(config: str, override: tuple[str, ...]) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        overrides = {}
        for item in override:
            k, _, v = item.partition("=")
            # Attempt numeric coercion
            try:
                overrides[k.strip()] = int(v)
            except ValueError:
                try:
                    overrides[k.strip()] = float(v)
                except ValueError:
                    overrides[k.strip()] = v
        run_generation(config, overrides=overrides)

    _cli()
