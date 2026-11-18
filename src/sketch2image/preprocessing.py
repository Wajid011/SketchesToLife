"""Sketch preprocessing — edge detection and conditioning-map generation.

Converts a raw hand-drawn sketch into a conditioning map suitable for
ControlNet or T2I-Adapter. Supported preprocessors:

- ``hed``             : HED (Holistically-nested Edge Detection) blob detector,
                        best for rough / freehand scribbles.
- ``pidinet``         : PiDiNet soft-edge detector, best for soft / watercolour
                        style sketches.
- ``lineart``         : Clean lineart extractor, best for clean line drawings.
- ``lineart_coarse``  : Coarser lineart variant, more robust to noise.
- ``none``            : No preprocessing; sketch used as-is.

All preprocessors are loaded from ``controlnet-aux`` and are therefore only
available when running in the full GPU environment (not locally).

.. note::
    Import guards ensure that ``controlnet_aux`` is optional at import time
    so that unit tests can run without it installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Preprocessor registry — populated lazily to avoid import errors on CPU
_PREPROCESSOR_REGISTRY: dict[str, type] = {}


def _load_preprocessor_classes() -> None:
    """Lazily import controlnet_aux classes into the registry.

    Raises:
        ImportError: If ``controlnet-aux`` is not installed.
    """
    global _PREPROCESSOR_REGISTRY  # noqa: PLW0603

    if _PREPROCESSOR_REGISTRY:
        return  # already loaded

    try:
        from controlnet_aux import HEDdetector, LineartDetector, PidiNetDetector

        _PREPROCESSOR_REGISTRY = {
            "hed": HEDdetector,
            "pidinet": PidiNetDetector,
            "lineart": LineartDetector,
            "lineart_coarse": LineartDetector,  # uses coarse=True kwarg
        }
        logger.debug("controlnet_aux preprocessors loaded successfully.")
    except ImportError as exc:
        raise ImportError(
            "controlnet-aux is required for sketch preprocessing. "
            "Install it with: pip install controlnet-aux>=0.0.7"
        ) from exc


class SketchPreprocessor:
    """Wraps a controlnet-aux preprocessor with a uniform interface.

    Args:
        preprocessor_type: One of ``hed``, ``pidinet``, ``lineart``,
            ``lineart_coarse``, or ``none``.
        resolution: Target resolution for the output conditioning map.

    Example:
        >>> prep = SketchPreprocessor("hed", resolution=512)
        >>> cond_map = prep(Image.open("sketch.png"))
    """

    def __init__(self, preprocessor_type: str, resolution: int = 512) -> None:
        self.preprocessor_type = preprocessor_type
        self.resolution = resolution
        self._detector = None  # loaded on first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply the preprocessor to a PIL image.

        Args:
            image: Input sketch as a PIL Image (RGB or grayscale).

        Returns:
            Conditioning map as a PIL Image (RGB, same resolution as input
            after preprocessing).
        """
        if self.preprocessor_type == "none":
            logger.debug("Preprocessor type is 'none'; returning sketch as-is.")
            return image.convert("RGB")

        detector = self._get_detector()
        logger.debug(
            "Applying '%s' preprocessor at resolution %d.",
            self.preprocessor_type,
            self.resolution,
        )

        kwargs: dict = {"detect_resolution": self.resolution, "image_resolution": self.resolution}
        if self.preprocessor_type == "lineart_coarse":
            kwargs["coarse"] = True

        result = detector(image, **kwargs)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_detector(self) -> object:
        """Return (and cache) the underlying detector instance.

        Returns:
            Instantiated controlnet-aux detector object.
        """
        if self._detector is None:
            _load_preprocessor_classes()
            cls = _PREPROCESSOR_REGISTRY[self.preprocessor_type]
            logger.info("Loading '%s' detector from pretrained weights.", self.preprocessor_type)
            self._detector = cls.from_pretrained("lllyasviel/Annotators")
        return self._detector


def load_sketch(path: str | Path, resolution: int | None = None) -> Image.Image:
    """Load a sketch from disk and optionally resize it.

    Args:
        path: Path to the sketch image file.
        resolution: If provided, resize the longest edge to this value
            while preserving aspect ratio.

    Returns:
        Sketch as an RGB PIL Image.

    Raises:
        FileNotFoundError: If the sketch file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sketch not found: {path}")

    image = Image.open(path).convert("RGB")
    logger.debug("Loaded sketch '%s' (original size: %s).", path.name, image.size)

    if resolution is not None:
        image = _resize_keep_aspect(image, resolution)
        logger.debug("Resized sketch to %s.", image.size)

    return image


def _resize_keep_aspect(image: Image.Image, max_side: int) -> Image.Image:
    """Resize an image so its longest side equals ``max_side``.

    Args:
        image: Input PIL image.
        max_side: Target length for the longest edge.

    Returns:
        Resized PIL image.
    """
    w, h = image.size
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Snap to multiples of 8 (required by diffusion VAE)
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    return image.resize((new_w, new_h), Image.LANCZOS)
