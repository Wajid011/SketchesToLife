#!/usr/bin/env python3
"""scripts/check_env.py — Environment diagnostic for SketchesToLife.

Run locally (CPU) to verify your dev setup.
Run on Colab (GPU) to confirm the backend can handle your VRAM.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --json   # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure src/ is on the path when running as a script (not installed)
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def _banner(text: str) -> None:
    width = 60
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def _row(label: str, value: object, ok: bool | None = None) -> None:
    indicator = ""
    if ok is True:
        indicator = "  ✓"
    elif ok is False:
        indicator = "  ✗"
    print(f"  {label:<32} {value}{indicator}")


def main(as_json: bool = False) -> None:
    """Collect and display environment info.

    Args:
        as_json: If True, print raw JSON instead of a human-readable table.
    """
    # Import lazily so the script works even if sketch2image is not installed
    try:
        from sketch2image.utils import get_env_info

        info = get_env_info()
    except ImportError:
        # Fallback: gather info directly (useful before `pip install -e .`)
        info = _fallback_env_info()

    if as_json:
        print(json.dumps(info, indent=2))
        return

    # ── Human-readable report ──────────────────────────────────────────
    _banner("SketchesToLife — Environment Check")
    print()

    print("  Python & PyTorch")
    print("  " + "-" * 40)
    _row("Python version", info.get("python", "unknown"))

    torch_ver = info.get("torch")
    _row("PyTorch", torch_ver or "NOT installed", ok=torch_ver is not None)

    cuda_ok = info.get("cuda_available", False)
    _row("CUDA available", cuda_ok, ok=cuda_ok)

    if cuda_ok:
        _row("CUDA version", info.get("cuda_version"))
        _row("GPU name", info.get("gpu_name"))
        vram = info.get("gpu_vram_gb", 0.0)
        vram_ok = vram >= 8.0
        _row("GPU VRAM", f"{vram} GB", ok=vram_ok)
    else:
        print()
        print("  ⚠  No CUDA GPU detected.")
        print("  ⚠  Inference must run on Google Colab (T4) or Kaggle.")

    print()
    print("  HuggingFace Libraries")
    print("  " + "-" * 40)
    for lib in ("diffusers", "transformers", "accelerate", "controlnet_aux"):
        ver = info.get(lib)
        _row(lib, ver or "NOT installed", ok=ver is not None)

    print()
    print("  Image & Utility Libraries")
    print("  " + "-" * 40)
    _check_import("PIL (Pillow)", "PIL")
    _check_import("numpy", "numpy")
    _check_import("cv2 (opencv-python)", "cv2")
    _check_import("matplotlib", "matplotlib")
    _check_import("yaml (pyyaml)", "yaml")
    _check_import("gradio", "gradio")
    _check_import("lpips", "lpips")
    _check_import("skimage (scikit-image)", "skimage")

    print()
    _banner("Recommendation")
    rec = info.get("recommended_backend", "unknown")
    print(f"\n  Recommended backend:  {rec}")

    if not cuda_ok:
        print()
        print("  Next steps for Colab:")
        print("  1. Push this repo to GitHub.")
        print("  2. Open notebooks/colab_runner.ipynb on Google Colab.")
        print("  3. The notebook will clone the repo, install deps,")
        print("     and run inference on the free T4 GPU.")
    else:
        vram = info.get("gpu_vram_gb", 0.0)
        if vram < 10:
            print()
            print("  Tip: Enable attention slicing in configs/default.yaml")
            print("       (enable_attention_slicing: true) to reduce VRAM usage.")
        elif vram >= 16:
            print()
            print("  Tip: You have enough VRAM for SDXL with cpu offload.")
            print("       Try backend: sdxl_controlnet in your config.")

    print()


def _check_import(label: str, module: str) -> None:
    """Print a row for an optional library import check.

    Args:
        label: Display label shown in the left column.
        module: Module name to try importing.
    """
    try:
        mod = __import__(module)
        ver = getattr(mod, "__version__", "installed")
        _row(label, ver, ok=True)
    except ImportError:
        _row(label, "NOT installed", ok=False)


def _fallback_env_info() -> dict:
    """Minimal env info gathered without the sketch2image package.

    Returns:
        Partial info dict.
    """
    import sys

    info: dict = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu_vram_gb"] = round(vram, 1)
        else:
            info.update(cuda_version=None, gpu_name=None, gpu_vram_gb=0.0)
    except ImportError:
        info.update(
            torch=None,
            cuda_available=False,
            cuda_version=None,
            gpu_name=None,
            gpu_vram_gb=0.0,
        )

    cuda_ok = info.get("cuda_available", False)
    vram = info.get("gpu_vram_gb", 0.0)
    if not cuda_ok:
        info["recommended_backend"] = "cpu_only — cannot run inference locally"
    elif vram >= 20:
        info["recommended_backend"] = "sdxl_controlnet (≥20 GB VRAM)"
    elif vram >= 10:
        info["recommended_backend"] = "sd15_controlnet (≥10 GB VRAM)"
    else:
        info["recommended_backend"] = "sd15_controlnet with attention slicing"

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SketchesToLife environment checker.")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print machine-readable JSON.",
    )
    args = parser.parse_args()
    main(as_json=args.as_json)
