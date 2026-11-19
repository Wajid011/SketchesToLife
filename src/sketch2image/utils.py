"""Shared utilities: config loading, path resolution, seed management.

Keeps cross-cutting concerns out of the domain modules so each module
stays focused on a single responsibility.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML config file, merging it on top of the defaults.

    Resolution order (later wins):
    1. ``configs/default.yaml`` (always loaded as base)
    2. ``config_path`` (experiment-specific YAML, if provided)
    3. ``overrides`` dict (runtime key=value pairs)

    Args:
        config_path: Optional path to an experiment YAML config.
        overrides: Optional dict of key-value overrides applied last.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If ``config_path`` is provided but does not exist.
    """
    # 1. Load defaults
    config: dict[str, Any] = {}
    if _DEFAULT_CONFIG_PATH.exists():
        with _DEFAULT_CONFIG_PATH.open() as f:
            config = yaml.safe_load(f) or {}
        logger.debug("Loaded default config from %s.", _DEFAULT_CONFIG_PATH)
    else:
        logger.warning(
            "Default config not found at %s; starting from empty.", _DEFAULT_CONFIG_PATH
        )

    # 2. Merge experiment config
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open() as f:
            experiment = yaml.safe_load(f) or {}
        config = _deep_merge(config, experiment)
        logger.debug("Merged experiment config from %s.", config_path)

    # 3. Apply runtime overrides
    if overrides:
        config = _deep_merge(config, overrides)
        logger.debug("Applied %d override(s).", len(overrides))

    return config


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base``.

    Args:
        base: Base dictionary (not mutated).
        overlay: Dictionary whose values take precedence.

    Returns:
        New merged dictionary.
    """
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """Resolve the output directory, preferring the ``OUTPUT_DIR`` env var.

    Resolution order:
    1. ``OUTPUT_DIR`` environment variable (set by Colab runner).
    2. ``config["output_dir"]`` from the YAML.
    3. Fallback: ``outputs/`` relative to the working directory.

    The resolved path is created if it does not exist.

    Args:
        config: Merged experiment configuration.

    Returns:
        Resolved and created :class:`pathlib.Path` to the output directory.
    """
    env_dir = os.environ.get("OUTPUT_DIR")
    if env_dir:
        path = Path(env_dir)
        logger.info("Output directory from OUTPUT_DIR env var: %s", path)
    elif "output_dir" in config:
        path = Path(config["output_dir"])
        logger.info("Output directory from config: %s", path)
    else:
        path = Path("outputs")
        logger.warning("No output_dir in config or env; defaulting to '%s'.", path)

    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility.

    Sets Python ``random``, ``numpy``, and ``torch`` (CPU + CUDA) seeds.
    Torch CUDA operations are made deterministic where possible.

    Args:
        seed: Integer seed value.
    """
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug("Seeds set to %d (torch + numpy + random).", seed)
    except ImportError:
        logger.debug("torch not available; seed set for random + numpy only.")


# ---------------------------------------------------------------------------
# Environment info (also called by scripts/check_env.py)
# ---------------------------------------------------------------------------


def get_env_info() -> dict[str, Any]:
    """Collect runtime environment information.

    Returns:
        Dictionary with keys: ``python``, ``torch``, ``cuda_available``,
        ``cuda_version``, ``gpu_name``, ``gpu_vram_gb``, ``diffusers``,
        ``transformers``, ``recommended_backend``.
    """
    import sys

    info: dict[str, Any] = {"python": sys.version.split()[0]}

    # Torch
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
            info["cuda_version"] = None
            info["gpu_name"] = None
            info["gpu_vram_gb"] = 0.0
    except ImportError:
        info.update(
            torch=None,
            cuda_available=False,
            cuda_version=None,
            gpu_name=None,
            gpu_vram_gb=0.0,
        )

    # HF libraries
    for lib in ("diffusers", "transformers", "accelerate", "controlnet_aux"):
        try:
            mod = __import__(lib)
            info[lib] = getattr(mod, "__version__", "installed")
        except ImportError:
            info[lib] = None

    # Backend recommendation
    vram = info.get("gpu_vram_gb", 0.0)
    if not info.get("cuda_available"):
        info["recommended_backend"] = "cpu_only — cannot run inference locally"
    elif vram >= 20:
        info["recommended_backend"] = "sdxl_controlnet (≥20 GB VRAM)"
    elif vram >= 10:
        info["recommended_backend"] = "sd15_controlnet (≥10 GB VRAM)"
    else:
        info["recommended_backend"] = (
            "sd15_controlnet with attention slicing (low VRAM)"
        )

    return info


def check_env_main() -> None:
    """CLI entry point for ``s2i-check-env`` (see pyproject.toml)."""
    import json

    info = get_env_info()
    print(json.dumps(info, indent=2))
