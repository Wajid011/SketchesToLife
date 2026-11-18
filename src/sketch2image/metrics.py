"""Image quality metrics for evaluating sketch-to-image fidelity.

Provides lightweight wrappers around common perceptual metrics.
All metrics are optional imports — the module stays importable on CPU
without torch / lpips installed (functions will raise ``ImportError``
with a clear message if called without the required dependency).

Metrics implemented:
- :func:`compute_ssim`   : Structural Similarity Index (scikit-image)
- :func:`compute_lpips`  : Learned Perceptual Image Patch Similarity
- :func:`edge_similarity`: Canny-edge-based structural overlap score
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def compute_ssim(image_a: Image.Image, image_b: Image.Image) -> float:
    """Compute the Structural Similarity Index between two images.

    Both images are converted to grayscale and resized to the smaller
    of the two before comparison.

    Args:
        image_a: First PIL image.
        image_b: Second PIL image.

    Returns:
        SSIM score in ``[0, 1]`` (1 = identical).

    Raises:
        ImportError: If ``scikit-image`` is not installed.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError as exc:
        raise ImportError("scikit-image is required for SSIM. pip install scikit-image") from exc

    a = np.array(image_a.convert("L"))
    b = np.array(image_b.convert("L"))

    # Align sizes
    if a.shape != b.shape:
        target = (min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1]))
        from PIL import Image as PILImage

        a = np.array(image_a.convert("L").resize((target[1], target[0]), PILImage.LANCZOS))
        b = np.array(image_b.convert("L").resize((target[1], target[0]), PILImage.LANCZOS))

    score: float = ssim(a, b, data_range=255)
    logger.debug("SSIM: %.4f", score)
    return score


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------


def compute_lpips(image_a: Image.Image, image_b: Image.Image, net: str = "alex") -> float:
    """Compute LPIPS perceptual distance between two images.

    Lower is better (0 = identical). Uses the ``lpips`` package.

    Args:
        image_a: First PIL image (RGB).
        image_b: Second PIL image (RGB).
        net: Backbone network to use — ``"alex"`` (default) or ``"vgg"``.

    Returns:
        LPIPS distance as a Python float.

    Raises:
        ImportError: If ``lpips`` or ``torch`` is not installed.
    """
    try:
        import lpips
        import torch
    except ImportError as exc:
        raise ImportError("lpips and torch are required. pip install lpips torch") from exc

    _loss_fn = _get_lpips_fn(net)

    def _to_tensor(img: Image.Image) -> Any:
        arr = np.array(img.convert("RGB").resize((256, 256))) / 127.5 - 1.0
        return torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        dist = _loss_fn(_to_tensor(image_a), _to_tensor(image_b))
    score = float(dist.item())
    logger.debug("LPIPS (%s): %.4f", net, score)
    return score


_lpips_cache: dict[str, Any] = {}


def _get_lpips_fn(net: str) -> Any:
    """Return (and cache) an LPIPS loss function for a given backbone.

    Args:
        net: Network name (``"alex"`` or ``"vgg"``).

    Returns:
        Cached ``lpips.LPIPS`` instance.
    """
    if net not in _lpips_cache:
        import lpips

        _lpips_cache[net] = lpips.LPIPS(net=net)
    return _lpips_cache[net]


# ---------------------------------------------------------------------------
# Edge similarity (structural overlap)
# ---------------------------------------------------------------------------


def edge_similarity(sketch: Image.Image, generated: Image.Image) -> float:
    """Measure structural overlap between a sketch and a generated image.

    Uses Canny edge detection on the generated image and computes the
    Dice coefficient against the binarised sketch edges.

    This is a lightweight proxy for "how well does the output follow the
    sketch structure" — useful for the experimentation section.

    Args:
        sketch: Original hand-drawn sketch (PIL Image).
        generated: Generated output image (PIL Image).

    Returns:
        Dice similarity coefficient in ``[0, 1]``.

    Raises:
        ImportError: If ``opencv-python`` is not installed.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required. pip install opencv-python") from exc

    def _edges(img: Image.Image) -> np.ndarray:
        gray = np.array(img.convert("L"))
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)

    target_size = (512, 512)
    sk_edges = _edges(sketch.resize(target_size, Image.LANCZOS))
    gen_edges = _edges(generated.resize(target_size, Image.LANCZOS))

    # Binarise
    sk_bin = (sk_edges > 0).astype(np.float32)
    gen_bin = (gen_edges > 0).astype(np.float32)

    intersection = (sk_bin * gen_bin).sum()
    union = sk_bin.sum() + gen_bin.sum()
    score = float(2.0 * intersection / (union + 1e-8))
    logger.debug("Edge similarity (Dice): %.4f", score)
    return score


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------


def compute_all_metrics(
    sketch: Image.Image,
    generated: Image.Image,
) -> dict[str, float]:
    """Compute all available metrics between a sketch and generated image.

    Gracefully skips metrics whose dependencies are not installed.

    Args:
        sketch: Original sketch.
        generated: Generated output.

    Returns:
        Dict mapping metric name to score.
    """
    results: dict[str, float] = {}

    try:
        results["ssim"] = compute_ssim(sketch, generated)
    except ImportError:
        logger.warning("scikit-image not available; skipping SSIM.")

    try:
        results["edge_similarity"] = edge_similarity(sketch, generated)
    except ImportError:
        logger.warning("opencv-python not available; skipping edge_similarity.")

    try:
        results["lpips_alex"] = compute_lpips(sketch, generated, net="alex")
    except ImportError:
        logger.warning("lpips not available; skipping LPIPS.")

    return results
