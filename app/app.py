#!/usr/bin/env python3
"""SketchesToLife — Gradio web app.

Upload or draw a sketch, see the control map the model will receive, and turn it
into an image with SD 1.5 + ControlNet (or SDXL / T2I-Adapter).

Run::

    python app/app.py            # http://127.0.0.1:7860
    python app/app.py --share    # public *.gradio.live link (use this on Colab)
    python app/app.py --mock     # force CPU placeholder mode (UI development)

Without a CUDA GPU the pipeline runs in *mock* mode: the UI is fully functional
but results are placeholders (the control map over a gradient), clearly labelled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import logging
import os
import random
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

# Handlers run in worker threads; GUI backends (TkAgg on Windows) crash the
# server with "Tcl_AsyncDelete: async handler deleted by the wrong thread".
matplotlib.use("Agg")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sketch2image.generate import STYLE_PRESETS, apply_style_preset  # noqa: E402
from sketch2image.pipelines import SketchToImageGenerator  # noqa: E402
from sketch2image.preprocessing import prepare_condition  # noqa: E402
from sketch2image.visualize import save_side_by_side  # noqa: E402

logger = logging.getLogger("app")

SKETCH_DIR = Path(os.environ.get("SKETCH_DIR", ROOT / "sketches"))
APP_OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs")) / "app"

BACKENDS = {
    "SD 1.5 + ControlNet (recommended)": "sd15_controlnet",
    "SDXL + ControlNet scribble (1024 px, slower)": "sdxl_controlnet",
    "SD 1.5 + T2I-Adapter sketch (lightweight)": "t2i_adapter",
}
CN_TYPES = {
    "sd15_controlnet": ["scribble", "lineart", "softedge"],
    "sdxl_controlnet": ["scribble"],
    "t2i_adapter": ["sketch"],
}
PREPROCESS_MODES = {
    "Cleaned sketch (recommended)": "clean",
    "Canny edges": "canny",
    "HED soft edges": "hed",
    "PiDiNet soft edges": "pidinet",
    "Neural lineart": "lineart",
    "Raw (no cleaning)": "none",
}
NEURAL_MODES = {"hed", "pidinet", "lineart"}
PRESET_CHOICES = ["none"] + list(STYLE_PRESETS)
MAX_SEED = 2**31 - 1

EXAMPLES = [
    [
        "sketch_01.png",
        "cozy wooden cabin in the rocky mountains, warm glowing windows, pine forest",
        "photorealistic",
    ],
    [
        "sketch_02.png",
        "futuristic sleek electric sports car, reflections on glossy metallic red paint",
        "concept_art",
    ],
    [
        "sketch_03.png",
        "cute distinguished gentleman cat wearing retro glasses and bowtie",
        "watercolor",
    ],
]

HOW_IT_WORKS = """
**1 · Preprocessing.** Your sketch is cleaned (illumination normalised, thresholded,
specks removed) and turned into a *control map*: white lines on black, padded to a
multiple of 64 px. That map — shown next to your input — is literally what the model sees.

**2 · ControlNet.** Stable Diffusion's UNet denoises a random latent while
cross-attending to your prompt. ControlNet is a *trainable copy* of the UNet's encoder
that reads the control map. Its features are added back into the frozen UNet through
**zero convolutions** — 1×1 convs initialised to zero, so training started from
"no influence" and learned only the structure signal. The base model is never changed.

**3 · Knobs.**
- **Conditioning scale** multiplies that injected signal: low → the sketch is a loose
  hint (lines get ignored); high → lines are followed rigidly and can be rendered as
  dark seams or flat outlines. 0.8–1.2 is the usual sweet spot.
- **Control end** switches ControlNet off for the last part of denoising, so fine
  texture is rendered freely once the layout is fixed.
- **Guidance scale** (CFG) controls how strongly the *prompt* is followed.

**Backends.** SDXL renders at 1024 px with more detail but is much slower and is CPU-offloaded
to fit 16 GB. T2I-Adapter is a ~77 M-parameter alternative to ControlNet — faster, looser.
"""

CSS = """
#title {margin-bottom: 0}
#mode-badge {font-size: 0.9em}
.compact-row {gap: 8px}
footer {display: none !important}
"""


# ---------------------------------------------------------------------------
# Model cache — one pipeline resident at a time (a T4 cannot hold two)
# ---------------------------------------------------------------------------


class ModelManager:
    """Keeps the most recently used pipeline loaded and reuses it across requests."""

    def __init__(self, force_mock: bool = False) -> None:
        self.force_mock = force_mock
        self._gen: SketchToImageGenerator | None = None
        self._lock = threading.Lock()

    @staticmethod
    def has_cuda() -> bool:
        """Whether a CUDA device is usable (False when torch is missing)."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def mock(self) -> bool:
        """True when results will be CPU placeholders rather than diffusion output."""
        return self.force_mock or not self.has_cuda()

    def get(self, backend: str, cn_type: str) -> SketchToImageGenerator:
        """Return a loaded generator, swapping models only when the choice changes."""
        with self._lock:
            g = self._gen
            if g is None or g.backend != backend or g.controlnet_type != cn_type:
                if g is not None:
                    g.unload()
                self._gen = SketchToImageGenerator(
                    backend=backend,
                    controlnet_type=cn_type,
                    # SDXL's loader always CPU-offloads by itself. Leaving this flag
                    # off keeps the noise generator on the GPU, exactly as in the
                    # CLI and experiments, so a seed gives the same image everywhere.
                    mock=self.force_mock,
                )
                self._gen.load()
            return self._gen

    def release(self) -> None:
        """Unload the resident pipeline and free its GPU memory (used after OOM)."""
        with self._lock:
            if self._gen is not None:
                self._gen.unload()
                self._gen = None


MODELS = ModelManager()


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def flatten_on_white(img: Image.Image) -> Image.Image:
    """Composite a (possibly transparent) image onto white and return RGB.

    The Sketchpad canvas is transparent; a naive RGBA->RGB conversion turns it
    black, which would make black strokes invisible to the preprocessor.
    """
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, img).convert("RGB")
    return img.convert("RGB")


def resolve_sketch(
    source: str, upload: Image.Image | None, drawing: Any
) -> Image.Image | None:
    """Pick the sketch from the active input tab.

    Args:
        source: ``"upload"`` or ``"draw"``.
        upload: Value of the upload component.
        drawing: Sketchpad value (dict with ``composite``) or image.

    Returns:
        RGB sketch, or ``None`` if the active input is empty.
    """
    if source == "draw":
        img = drawing.get("composite") if isinstance(drawing, dict) else drawing
    else:
        img = upload
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    rgb = flatten_on_white(img)
    arr = np.asarray(rgb.convert("L"))
    if arr.min() > 245 or arr.max() < 10:  # blank page
        return None
    return rgb


def _resolution(backend: str) -> int:
    return 1024 if backend == "sdxl_controlnet" else 512


def preview_condition(
    source: str, upload: Any, drawing: Any, mode_label: str, thickness: int
) -> tuple[Image.Image | None, str]:
    """Render the control map the model will receive (always at 512 px for speed)."""
    sketch = resolve_sketch(source, upload, drawing)
    if sketch is None:
        return None, ""
    mode = PREPROCESS_MODES[mode_label]
    note = ""
    if mode in NEURAL_MODES and importlib.util.find_spec("controlnet_aux") is None:
        note = (
            "⚠️ `controlnet-aux` is not installed here, so this mode falls back to the "
            "cleaned sketch."
        )
    cmap = prepare_condition(
        sketch, mode=mode, resolution=512, line_thickness=int(thickness)
    )
    return cmap, note


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    return "out of memory" in str(exc).lower()


def _oom_message(backend: str, n_images: int) -> str:
    tips = []
    if n_images > 1:
        tips.append(f"generate 1 image instead of {n_images}")
    if backend == "sdxl_controlnet":
        tips.append(
            "switch to SD 1.5 (512 px uses far less memory than SDXL at 1024 px)"
        )
    tips += ["lower the step count", "close other GPU notebooks/sessions and retry"]
    return "GPU ran out of memory. Try: " + "; ".join(tips) + "."


def generate(
    source: str,
    upload: Any,
    drawing: Any,
    prompt: str,
    preset: str,
    negative: str,
    backend_label: str,
    cn_type: str,
    scale: float,
    cfg: float,
    steps: int,
    control_end: float,
    seed: int,
    randomize: bool,
    n_images: int,
    mode_label: str,
    thickness: int,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects progress this way
) -> tuple[list[tuple[Image.Image, str]], str, str, str, int]:
    """Gradio handler: run the pipeline and package results.

    Returns:
        (gallery items, side-by-side path, run summary markdown, zip path, seed used)
    """
    sketch = resolve_sketch(source, upload, drawing)
    problem = None
    if sketch is None:
        problem = "No sketch yet — upload an image or draw on the canvas first."
    elif not prompt or not prompt.strip():
        problem = "Please describe what the sketch should become in the prompt box."
    if problem:
        # Warning toast + no-op outputs: gr.Error would wipe the previous results.
        gr.Warning(problem)
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    backend = BACKENDS[backend_label]
    mode = PREPROCESS_MODES[mode_label]
    seed = random.randint(0, MAX_SEED) if randomize else int(seed)
    res = _resolution(backend)
    styled, full_negative = apply_style_preset(
        prompt, None if preset == "none" else preset, negative or ""
    )

    progress(0.05, desc="Loading model (first run per backend downloads weights)…")
    try:
        gen = MODELS.get(backend, cn_type)
        progress(0.3, desc=f"Generating {int(n_images)} image(s)…")
        images, meta = gen.generate(
            sketch=sketch,
            prompt=styled,
            negative_prompt=full_negative,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            controlnet_conditioning_scale=float(scale),
            control_guidance_end=float(control_end),
            resolution=res,
            seed=seed,
            num_images_per_prompt=int(n_images),
            condition_mode=mode,
            condition_kwargs={"line_thickness": int(thickness)},
        )
    except Exception as exc:  # noqa: BLE001 - surface every failure in the UI
        if _is_oom(exc):
            MODELS.release()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
            raise gr.Error(_oom_message(backend, int(n_images))) from exc
        logger.exception("Generation failed")
        raise gr.Error(f"Generation failed: {type(exc).__name__}: {exc}") from exc

    progress(0.9, desc="Saving results…")
    cmap = prepare_condition(
        sketch, mode=mode, resolution=res, line_thickness=int(thickness)
    )
    run_dir = APP_OUTPUT_DIR / dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    sketch.save(run_dir / "sketch.png")
    cmap.save(run_dir / "control_map.png")
    files = []
    for i, img in enumerate(images):
        img.save(run_dir / f"result_{i}.png")
        files.append(f"result_{i}.png")
    side = save_side_by_side(
        sketch,
        cmap,
        images[0],
        run_dir / "side_by_side.png",
        titles=(
            "Your sketch",
            "Control map (what the model sees)",
            "Result" + (" — MOCK" if meta.get("mock") else ""),
        ),
    )
    meta.update(
        {
            "raw_prompt": prompt,
            "style_preset": None if preset == "none" else preset,
            "line_thickness": int(thickness),
            "generated_files": files,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    (run_dir / "settings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    zip_path = run_dir / f"sketchestolife_seed{seed}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in [
            "sketch.png",
            "control_map.png",
            "side_by_side.png",
            "settings.json",
            *files,
        ]:
            zf.write(run_dir / f, f)

    vram = meta.get("peak_vram_mb", 0.0)
    summary = (
        f"**{len(images)} image(s)** · {res}×{res} · seed `{seed}` · "
        f"{meta.get('runtime_sec', 0):.1f} s"
        + (f" · peak VRAM {vram / 1024:.1f} GB" if vram else "")
        + (
            "  \n⚠️ **Mock mode** — placeholder output, no diffusion model was run."
            if meta.get("mock")
            else ""
        )
    )
    gallery = [(img, f"#{i + 1} · seed {seed}") for i, img in enumerate(images)]
    return gallery, str(side), summary, str(zip_path), seed


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def _mode_badge() -> str:
    if MODELS.mock:
        return (
            "🟡 **Mock mode** — no CUDA GPU detected. The interface works end-to-end, "
            "but results are placeholders. Run on Colab/GPU for real images."
        )
    import torch

    return f"🟢 **GPU:** {torch.cuda.get_device_name(0)}"


def build_ui() -> gr.Blocks:
    """Assemble the Blocks interface."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )
    with gr.Blocks(title="SketchesToLife", theme=theme, css=CSS) as demo:
        gr.Markdown(
            "# ✏️ SketchesToLife\nTurn a hand-drawn sketch into an image while "
            "keeping its layout — Stable Diffusion + ControlNet.",
            elem_id="title",
        )
        gr.Markdown(_mode_badge(), elem_id="mode-badge")
        source = gr.State("upload")
        drawn = gr.State(None)  # flattened Sketchpad image (see wiring below)

        with gr.Row(equal_height=False):
            # ---------------- left: inputs ----------------
            with gr.Column(scale=5):
                with gr.Row(elem_classes="compact-row"):
                    with gr.Column(scale=1, min_width=240):
                        with gr.Tabs():
                            with gr.Tab("Upload", id="upload") as tab_up:
                                upload = gr.Image(
                                    label="Sketch image",
                                    type="pil",
                                    sources=["upload", "clipboard"],
                                    height=300,
                                )
                            with gr.Tab("Draw", id="draw") as tab_draw:
                                drawing = gr.Sketchpad(
                                    label="Draw here",
                                    type="pil",
                                    height=300,
                                    brush=gr.Brush(
                                        colors=["#000000"],
                                        default_size=4,
                                        color_mode="fixed",
                                    ),
                                    canvas_size=(512, 512),
                                )
                    with gr.Column(scale=1, min_width=240):
                        preview = gr.Image(
                            label="Control map — what the model sees",
                            interactive=False,
                            height=300,
                            show_download_button=False,
                        )
                        preview_note = gr.Markdown()

                prompt = gr.Textbox(
                    label="Prompt",
                    lines=2,
                    placeholder="e.g. cozy wooden cabin in the mountains, warm light",
                )
                with gr.Row():
                    preset = gr.Dropdown(
                        PRESET_CHOICES,
                        value="photorealistic",
                        label="Style preset",
                        scale=2,
                    )
                    run_btn = gr.Button(
                        "Generate", variant="primary", scale=1, size="lg"
                    )
                with gr.Accordion("Negative prompt", open=False):
                    negative = gr.Textbox(
                        label="Extra negative terms",
                        lines=2,
                        info="Added to a built-in list (blurry, watermark, bad anatomy, …) "
                        "and the preset's own negatives.",
                    )
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        backend = gr.Dropdown(
                            list(BACKENDS), value=list(BACKENDS)[0], label="Backend"
                        )
                        cn_type = gr.Dropdown(
                            CN_TYPES["sd15_controlnet"],
                            value="scribble",
                            label="ControlNet type",
                        )
                    with gr.Row():
                        scale = gr.Slider(
                            0.0,
                            2.0,
                            1.0,
                            step=0.05,
                            label="Conditioning scale (sketch strength)",
                        )
                        control_end = gr.Slider(
                            0.1,
                            1.0,
                            1.0,
                            step=0.05,
                            label="Control end (fraction of steps)",
                        )
                    with gr.Row():
                        cfg = gr.Slider(
                            1.0, 15.0, 7.5, step=0.5, label="Guidance scale (CFG)"
                        )
                        steps = gr.Slider(10, 60, 30, step=1, label="Steps")
                    with gr.Row():
                        seed = gr.Number(
                            42, label="Seed", precision=0, minimum=0, maximum=MAX_SEED
                        )
                        randomize = gr.Checkbox(False, label="Randomize seed")
                        n_images = gr.Slider(1, 4, 1, step=1, label="Images")
                    with gr.Row():
                        mode = gr.Dropdown(
                            list(PREPROCESS_MODES),
                            value=list(PREPROCESS_MODES)[0],
                            label="Preprocessing",
                        )
                        thickness = gr.Slider(
                            -2,
                            3,
                            0,
                            step=1,
                            label="Line thickness",
                            info="Thin (−) / thicken (+) strokes; cleaned mode only",
                        )

            # ---------------- right: outputs ----------------
            with gr.Column(scale=5):
                gallery = gr.Gallery(
                    label="Results",
                    columns=2,
                    height=360,
                    object_fit="contain",
                    preview=True,
                )
                summary = gr.Markdown()
                side = gr.Image(
                    label="Side by side: sketch · control map · result",
                    show_label=False,
                    interactive=False,
                )
                download = gr.DownloadButton(
                    "⬇ Download images + settings (.zip)",
                    interactive=True,
                    variant="secondary",
                )

        gr.Examples(
            examples=[
                [str(SKETCH_DIR / f), p, s]
                for f, p, s in EXAMPLES
                if (SKETCH_DIR / f).exists()
            ],
            inputs=[upload, prompt, preset],
            label="Example sketches (click to load)",
        )
        with gr.Accordion("How it works", open=False):
            gr.Markdown(HOW_IT_WORKS)

        # ---------------- wiring ----------------
        # Only drawing.change ever receives the raw Sketchpad value; it is flattened
        # into `drawn` (server-side State) and everything else reads that. Passing the
        # Sketchpad to several concurrent events makes Gradio rewrite/read the same
        # cached PNG at once, which fails on Windows with PermissionError.
        prev_inputs = [source, upload, drawn, mode, thickness]
        for tab, name in ((tab_up, "upload"), (tab_draw, "draw")):
            tab.select(lambda n=name: n, outputs=source).then(
                preview_condition,
                prev_inputs,
                [preview, preview_note],
                show_progress="hidden",
            )
        for comp in (upload, mode, thickness):
            comp.change(
                preview_condition,
                prev_inputs,
                [preview, preview_note],
                show_progress="hidden",
            )
        drawing.change(
            lambda d: resolve_sketch("draw", None, d),
            drawing,
            drawn,
            show_progress="hidden",
            trigger_mode="always_last",
        ).then(
            preview_condition,
            prev_inputs,
            [preview, preview_note],
            show_progress="hidden",
        )

        def _on_backend(label: str) -> dict:
            types = CN_TYPES[BACKENDS[label]]
            return gr.update(choices=types, value=types[0], interactive=len(types) > 1)

        backend.change(_on_backend, backend, cn_type)

        gen_inputs = [
            source,
            upload,
            drawn,
            prompt,
            preset,
            negative,
            backend,
            cn_type,
            scale,
            cfg,
            steps,
            control_end,
            seed,
            randomize,
            n_images,
            mode,
            thickness,
        ]
        gen_outputs = [gallery, side, summary, download, seed]
        run_btn.click(generate, gen_inputs, gen_outputs, concurrency_limit=1)
        prompt.submit(generate, gen_inputs, gen_outputs, concurrency_limit=1)
    return demo


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="SketchesToLife Gradio app")
    ap.add_argument(
        "--share", action="store_true", help="Create a public gradio.live link."
    )
    # Hugging Face Spaces / Docker set these env vars (0.0.0.0:7860); honour them.
    ap.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    )
    ap.add_argument(
        "--mock", action="store_true", help="Force placeholder mode (no GPU use)."
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    MODELS.force_mock = args.mock
    demo = build_ui()
    demo.queue(max_size=16, default_concurrency_limit=1)  # one GPU -> one job at a time
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(APP_OUTPUT_DIR), str(SKETCH_DIR)],
    )


if __name__ == "__main__":
    main()
