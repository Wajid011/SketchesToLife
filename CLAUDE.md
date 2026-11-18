# SketchesToLife — Project Context & Conventions

> **Last updated:** 2026-11-18  
> **Author:** Master's student, MSC-UPEC — 3rd Semester  
> **Purpose:** This file keeps every future AI coding session aligned with the
> project's decisions so we never have to re-explain them.

---

## 1. Project Overview

**Goal:** A sketch-to-image system that converts hand-drawn sketches into
realistic / visually compelling images while preserving the sketch's spatial
structure, using **pretrained diffusion models** wired together via a custom
Python pipeline — no online UI, no fine-tuning.

**Grading weights (keep these in mind when making trade-offs):**

| Criterion | Weight |
|---|---|
| Technical implementation & understanding | 30 % |
| Result quality & sketch preservation | 25 % |
| Experimentation & analysis | 20 % |
| Code quality / docs / reproducibility | 15 % |
| Creativity | 10 % |
| *(Bonus)* Optional Gradio web interface | extra |

**In scope:** ControlNet inference, preprocessor comparison, SDXL comparison,
T2I-Adapter comparison, Gradio UI, Colab notebook.  
**Out of scope (for now):** fine-tuning, chatbot, agent integrations.

---

## 2. Hardware & Execution Model

| Environment | Spec | Role |
|---|---|---|
| **Local laptop** | 8 GB RAM, no GPU, 256 GB SSD, Windows | Write code, run unit tests (CPU, mocked GPU) |
| **Google Colab** | Free T4, 16 GB VRAM | All diffusion inference |
| **Kaggle** | 2× T4, 30 GB VRAM | Fallback / SDXL runs |

**Golden rule:** *Never download a model or run diffusion locally.*  
Local tests MUST mock or `pytest.mark.skip` anything GPU-dependent.

The main execution entry point for Colab is:
`notebooks/colab_runner.ipynb` — it mounts Google Drive, does `git pull`,
installs requirements, then calls any experiment script with a chosen config.
Outputs land in `/content/drive/MyDrive/sketch2image/` (or `OUTPUT_DIR` env var).

---

## 3. Tech Stack

```
Python          3.10+
PyTorch         ≥ 2.1  (cu121 wheel on Colab/Kaggle)
diffusers       ≥ 0.27
transformers    ≥ 4.38
accelerate      ≥ 0.28
controlnet-aux  ≥ 0.0.7   (HED / PIDI / lineart preprocessors)
gradio          ≥ 4.x     (web UI, --share for Colab)
opencv-python   (preprocessing helpers)
Pillow, numpy, matplotlib, scipy
pyyaml          (config files)
pytest + ruff   (local dev only)
```

---

## 4. Verified Model IDs & Licenses

| Role | Hugging Face ID | License |
|---|---|---|
| **Base SD 1.5** | `sd-legacy/stable-diffusion-v1-5` | CreativeML Open RAIL-M |
| **ControlNet — Scribble (SD 1.5)** | `lllyasviel/control_v11p_sd15_scribble` | CreativeML Open RAIL-M |
| **ControlNet — Lineart (SD 1.5)** | `lllyasviel/control_v11p_sd15_lineart` | CreativeML Open RAIL-M |
| **ControlNet — Softedge (SD 1.5)** | `lllyasviel/control_v11p_sd15_softedge` | CreativeML Open RAIL-M |
| **Base SDXL** | `stabilityai/stable-diffusion-xl-base-1.0` | CreativeML Open RAIL++-M |
| **ControlNet Scribble (SDXL)** | `xinsir/controlnet-scribble-sdxl-1.0` | Apache-2.0 |
| **T2I-Adapter Sketch (SD 1.5)** | `TencentARC/t2iadapter_sketch_sd15v2` | Apache-2.0 |

> **Policy:** Never invent a model ID. If a new model is needed, search
> huggingface.co, confirm it exists, record its license here.

---

## 5. Architecture Decision Record

### 5.1 Primary backend — SD 1.5 + ControlNet
Main pipeline for all experiments and the report.  
Runs comfortably on a T4 at fp16 with attention slicing enabled.

### 5.2 Secondary backend — SDXL + ControlNet
Comparison only; must use `enable_model_cpu_offload()` and fp16 to fit in 16 GB.
Produces higher-resolution outputs (1024 × 1024).

### 5.3 T2I-Adapter comparison
Lightweight alternative to ControlNet; useful for the experimentation section.

### 5.4 Preprocessor options (ControlNet conditioning maps)
- **Scribble / HED** — best for rough freehand sketches  
- **Lineart** — best for clean line drawings  
- **Softedge (PIDI)** — best for soft / watercolour-style sketches  
Preprocessors live in `src/preprocessing/` and are selected via YAML config.

---

## 6. How ControlNet Conditions SD on the Sketch

*(Technical story — must appear in the report and be understood in the viva.)*

1. **Standard SD UNet** encodes a noisy latent and cross-attends to a text
   embedding to predict the noise.
2. **ControlNet** adds a **trainable copy of the SD encoder blocks**
   (down-blocks + middle block) alongside the frozen UNet.
3. The sketch (or its edge map) is passed through this trainable encoder.
   The encoder outputs are connected back into the frozen UNet decoder via
   **zero convolutions** — 1 × 1 convolutions initialised to zero weight *and*
   zero bias. At training start, zero convolutions output zero, so the ControlNet
   adds nothing: the base model is undisturbed. As training progresses, the
   convolutions learn to inject structure signal.
4. At inference, the sketch conditioning map is multiplied by a scalar
   **`conditioning_scale`** (typically 0.5–1.5) before being added to the UNet
   activations, letting us interpolate between "ignore sketch" and "follow sketch
   rigidly". This is our primary experiment knob.
5. The base UNet weights are **fully frozen**; only the ControlNet encoder and
   the zero convolutions are trained — keeping the original generation quality
   intact.

---

## 7. Repository Layout (Bird's-Eye)

```
SketchesToLife/
├── CLAUDE.md                  ← this file
├── README.md                  ← public-facing project README (to be written)
├── requirements.txt           ← Colab / GPU full deps
├── requirements-dev.txt       ← local CPU-only dev deps
├── .gitignore
│
├── configs/                   ← YAML experiment configs (one file per run)
│   ├── sd15_scribble_base.yaml
│   ├── sd15_lineart.yaml
│   ├── sd15_softedge.yaml
│   ├── sdxl_scribble.yaml
│   └── t2i_adapter_sketch.yaml
│
├── sketches/                  ← ≥ 3 hand-drawn input sketches (PNG/JPG)
│
├── outputs/                   ← gitignored; generated images land here locally
│
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── sd15_controlnet.py   ← SD 1.5 + ControlNet inference pipeline
│   │   ├── sdxl_controlnet.py   ← SDXL + ControlNet inference pipeline
│   │   └── t2i_adapter.py       ← T2I-Adapter inference pipeline
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── edge_detection.py    ← HED / PIDI / lineart preprocessors
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── visualise.py         ← side-by-side comparison grids
│   └── utils/
│       ├── __init__.py
│       ├── config.py            ← YAML loader + env-var output path resolver
│       └── seed.py              ← fixed-seed helpers
│
├── app/
│   └── gradio_app.py            ← Gradio web UI (upload OR draw, prompt, settings)
│
├── experiments/
│   └── run_experiment.py        ← CLI entry point; resumable (skip existing outputs)
│
├── notebooks/
│   └── colab_runner.ipynb       ← Colab main entry point
│
├── tests/
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_edge_detection.py
│   │   └── test_seed.py
│   └── integration/
│       └── test_pipeline_mock.py  ← mocked GPU pipeline tests
│
├── report/
│   └── report.md / report.pdf    ← 2–4 page technical report
│
└── assets/
    └── sample_outputs/            ← curated images for README / report
```

---

## 8. Code Conventions

### 8.1 Style
- **Formatter / linter:** `ruff` (line length 88). Run `ruff check . --fix` before commits.
- **Type hints:** all public functions and methods must have full type annotations
  (`from __future__ import annotations` at top of every module).
- **Docstrings:** Google-style (Args / Returns / Raises). Every public function.
- **Logging:** use `logging.getLogger(__name__)` — never `print()` in library code.
  Scripts / notebooks may use `print()` for user-facing output only.
- **No hardcoded paths:** all paths come from YAML config or env vars
  (`OUTPUT_DIR`, `SKETCH_DIR`). Use `pathlib.Path` everywhere.

### 8.2 Config-driven runs
Every experiment is fully described by a YAML file under `configs/`.  
Example mandatory keys: `model_id`, `controlnet_id`, `conditioning_scale`,
`num_inference_steps`, `guidance_scale`, `seed`, `output_dir`, `sketch_paths`.

### 8.3 Reproducibility
- Fixed seeds via `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)`.
- Record diffusers version + torch version in every output log.
- `run_experiment.py` skips images that already exist on disk (resumable runs).

### 8.4 fp16 & memory
- Always load models with `torch_dtype=torch.float16` on CUDA.
- Enable `pipe.enable_attention_slicing()` and
  `pipe.enable_model_cpu_offload()` for SDXL.
- For SD 1.5, `attention_slicing` is enough.

### 8.5 Testing locally
- Any test that needs a GPU must be decorated with
  `@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")`.
- Mock `diffusers` pipeline calls in unit tests with `unittest.mock.patch`.

---

## 9. Deliverables Checklist

- [ ] Source code on GitHub (public repo)
- [ ] ≥ 3 input sketches in `sketches/`
- [ ] Generated results (sketch ↔ output side-by-side, multiple prompts/settings)
- [ ] Technical explanation (this CLAUDE.md + report + inline docstrings)
- [ ] `README.md` with install, deps, model refs, settings
- [ ] `notebooks/colab_runner.ipynb` (full reproducible run)
- [ ] Full references section in `report/`
- [ ] 2–4 page technical report

---

## 10. Session Handoff Notes

When starting a new session, tell the AI:
> "Read CLAUDE.md. We are on task X."

The AI should then proceed without re-asking about stack, model IDs, or conventions.
