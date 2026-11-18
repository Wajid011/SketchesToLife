# SketchesToLife

**Bringing Sketches to Life with Generative AI** — MSc individual project.

A sketch-to-image pipeline that converts hand-drawn sketches into realistic images
using Stable Diffusion + ControlNet (and optional SDXL / T2I-Adapter backends),
all configurable via YAML and reproducible via fixed seeds.

> ⚠️ All heavy inference runs on **Google Colab** (free T4). See
> `notebooks/colab_runner.ipynb` for the main entry point.

## Quick Links
- [CLAUDE.md](CLAUDE.md) — full project context, conventions, model IDs, architecture
- [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb) — Colab runner

## Install (local dev, CPU only)
```bash
pip install -r requirements-dev.txt
pytest tests/unit/
```

## Run on Colab
Open `notebooks/colab_runner.ipynb` → Run All. It will mount Drive, pull the repo,
install `requirements.txt`, and execute an experiment with a chosen config.

## Models Used
See [CLAUDE.md § 4](CLAUDE.md#4-verified-model-ids--licenses) for the full table
of verified Hugging Face IDs and licenses.
