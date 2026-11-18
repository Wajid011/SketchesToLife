"""Sketch preprocessing — edge detection, cleaning, and conditioning-map generation.

Provides a robust pipeline for hand-drawn sketches (scans, photos, mobile uploads)
addressing real-world noise:
1. EXIF rotation correction and grayscale conversion.
2. Background illumination correction and normalisation.
3. Adaptive thresholding with automatic inversion detection (white lines on black).
4. Small-blob / speckle noise removal.
5. Morphological line thickness adjustment (dilation / erosion).
6. Aspect-ratio preserving resize with padding to multiples of 8 and 64 (512, 1024).
7. Multiple conditioning extractors: custom clean sketch, Canny edge detector,
   and neural detectors via ``controlnet-aux`` (HED, PiDiNet, Lineart).

Exposes:
- :func:`prepare_condition` : Unified function to process any sketch into a conditioning map.
- :func:`clean_sketch`      : Step-by-step cleaning of raw noisy sketches.
- :func:`resize_and_pad`    : Letterbox padding to target multiples of 8 / 64.
- :class:`SketchPreprocessor`: Class interface for backward compatibility with diffusers.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Registry for controlnet_aux detectors
_AUX_DETECTOR_REGISTRY: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 1. Image Loading & Orientation
# ---------------------------------------------------------------------------


def load_and_orient_image(image_input: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Load an image from path or object, fix EXIF orientation, and convert to grayscale.

    Args:
        image_input: File path, PIL Image, or numpy array.

    Returns:
        2D uint8 numpy array (grayscale, 0-255).

    Raises:
        FileNotFoundError: If a file path is provided but does not exist.
        ValueError: If input format is invalid.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at: {path}")
        pil_img = Image.open(path)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            # Assume RGB or BGR, convert to grayscale
            if image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2GRAY)
            return cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
        return image_input.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Correct EXIF rotation (e.g. smartphone photos)
    pil_img = ImageOps.exif_transpose(pil_img)

    # Convert to grayscale numpy array
    gray_img = pil_img.convert("L")
    return np.array(gray_img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 2. Cleaning Pipeline
# ---------------------------------------------------------------------------


def normalize_illumination(
    gray: np.ndarray,
    kernel_size: int = 25,
    clip_limit: float = 2.0,
) -> np.ndarray:
    """Correct non-uniform lighting/shadows in paper scans and photos.

    Divides the image by an estimated low-frequency background illumination
    followed by optional CLAHE contrast equalization.

    Args:
        gray: Grayscale input image (uint8, 0-255).
        kernel_size: Morphological kernel size for background estimation.
        clip_limit: CLAHE clip limit (set <= 0 to disable CLAHE).

    Returns:
        Illumination-normalised uint8 image (0-255).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Estimate background via morphological closing (fills dark strokes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Smooth the background estimation
    background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)

    # Division normalisation: (gray / background) * 255
    # Avoid zero division
    bg_float = np.maximum(background.astype(np.float32), 1.0)
    normalized = (gray.astype(np.float32) / bg_float) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    if clip_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        normalized = clahe.apply(normalized)

    return normalized


def auto_invert_to_white_lines(binary: np.ndarray) -> tuple[np.ndarray, bool]:
    """Ensure white lines (255) on black background (0).

    Detects whether the image is currently black-on-white (typical paper sketch)
    or white-on-black (chalkboard / inverted), and inverts if necessary.

    Args:
        binary: Binary image with values {0, 255} (or grayscale).

    Returns:
        Tuple of (inverted_or_original_array, was_inverted_bool).
    """
    # Sample border pixels to detect dominant background color
    h, w = binary.shape[:2]
    border_pixels = np.concatenate([
        binary[0, :],         # top
        binary[h - 1, :],     # bottom
        binary[:, 0],         # left
        binary[:, w - 1],     # right
    ])
    border_mean = np.mean(border_pixels)
    overall_mean = np.mean(binary)

    # If border or majority is light (> 127), background is white -> invert
    if border_mean > 127 or overall_mean > 127:
        inverted = cv2.bitwise_not(binary)
        return inverted, True
    return binary.copy(), False


def remove_small_blobs(binary: np.ndarray, min_area: int = 25) -> np.ndarray:
    """Remove small speckles/dust from a binary white-on-black image.

    Args:
        binary: Binary image with white foreground (255) on black background (0).
        min_area: Minimum pixel area to retain a connected component.

    Returns:
        Cleaned binary image.
    """
    if min_area <= 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = binary.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            cleaned[labels == i] = 0

    return cleaned


def adjust_line_thickness(binary: np.ndarray, line_thickness: int = 0) -> np.ndarray:
    """Thicken (dilate) or thin (erode) foreground strokes.

    Args:
        binary: Binary image with white foreground (255) on black (0).
        line_thickness: Positive to dilate (thicken), negative to erode (thin),
            0 for no change.

    Returns:
        Adjusted binary image.
    """
    if line_thickness == 0:
        return binary

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if line_thickness > 0:
        return cv2.dilate(binary, kernel, iterations=line_thickness)
    return cv2.erode(binary, kernel, iterations=abs(line_thickness))


def clean_sketch(
    image_input: str | Path | Image.Image | np.ndarray,
    adaptive_block_size: int = 25,
    adaptive_c: int = 10,
    min_blob_area: int = 20,
    line_thickness: int = 0,
    blur_kernel: int = 3,
) -> np.ndarray:
    """Execute the full sketch cleaning pipeline on noisy hand-drawn input.

    Steps:
    1. EXIF orientation + grayscale conversion.
    2. Illumination background normalization.
    3. Gaussian smoothing to suppress sensor grain.
    4. Adaptive Gaussian thresholding.
    5. Automatic inversion (guarantee white lines on black background).
    6. Small-blob / speckle noise removal.
    7. Morphological line thickness adjustment.

    Args:
        image_input: Input sketch.
        adaptive_block_size: Neighborhood size for adaptive thresholding (must be odd).
        adaptive_c: Constant subtracted from mean in adaptive thresholding.
        min_blob_area: Minimum area in pixels for small-blob removal.
        line_thickness: Morphological dilation (>0) or thinning (<0).
        blur_kernel: Gaussian blur kernel size (odd).

    Returns:
        Cleaned 2D binary uint8 image (255 for lines, 0 for background).
    """
    gray = load_and_orient_image(image_input)

    # 1. Illumination normalisation
    normalized = normalize_illumination(gray, kernel_size=31)

    # 2. Slight blur to reduce sensor noise
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        normalized = cv2.GaussianBlur(normalized, (blur_kernel, blur_kernel), 0)

    # 3. Adaptive thresholding
    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        adaptive_c,
    )

    # 4. Invert to ensure white lines on black background
    white_lines, _ = auto_invert_to_white_lines(binary)

    # 5. Remove small noise blobs / dust
    cleaned = remove_small_blobs(white_lines, min_area=min_blob_area)

    # 6. Adjust stroke thickness
    adjusted = adjust_line_thickness(cleaned, line_thickness=line_thickness)

    return adjusted


# ---------------------------------------------------------------------------
# 3. Aspect-Ratio Preserving Resize with Multiple-of-64 Padding
# ---------------------------------------------------------------------------


def resize_and_pad(
    image: np.ndarray | Image.Image,
    resolution: int | tuple[int, int] = 512,
    multiple: int = 64,
    pad_value: int = 0,
) -> np.ndarray:
    """Resize an image preserving aspect ratio with centered padding.

    Guarantees the output width and height are divisible by ``multiple``
    (default 64, suitable for Stable Diffusion 1.5 and SDXL VAE encoders).

    Args:
        image: 2D or 3D numpy array, or PIL Image.
        resolution: Target size. Either an integer (e.g. 512, 1024) for a square
            canvas, or a tuple ``(target_w, target_h)``.
        multiple: Required divisibility factor (default 64; 8 also valid).
        pad_value: Pixel value for padding canvas (default 0 = black).

    Returns:
        Padded uint8 numpy array with shape (target_h, target_w) or
        (target_h, target_w, channels).
    """
    if isinstance(image, Image.Image):
        arr = np.array(image)
    else:
        arr = image

    h, w = arr.shape[:2]

    if isinstance(resolution, int):
        target_w = (resolution // multiple) * multiple
        target_h = (resolution // multiple) * multiple
    else:
        target_w = (resolution[0] // multiple) * multiple
        target_h = (resolution[1] // multiple) * multiple

    target_w = max(target_w, multiple)
    target_h = max(target_h, multiple)

    # Compute aspect-preserving scale factor
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Choose interpolation: INTER_AREA for downscale, INTER_LINEAR for upscale
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)

    # Create padded canvas
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    if arr.ndim == 2:
        canvas = np.full((target_h, target_w), pad_value, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    else:
        channels = arr.shape[2]
        canvas = np.full((target_h, target_w, channels), pad_value, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w, :] = resized

    return canvas


# ---------------------------------------------------------------------------
# 4. Multi-Conditioning Extractors (Canny & controlnet_aux)
# ---------------------------------------------------------------------------


def extract_canny(
    image: np.ndarray | Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> np.ndarray:
    """Extract Canny edges from image (white edges on black background).

    Args:
        image: Grayscale or RGB image.
        low_threshold: Canny hysteresis lower bound.
        high_threshold: Canny hysteresis upper bound.

    Returns:
        2D uint8 binary array (0 or 255).
    """
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"))
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def _get_aux_detector(detector_type: str) -> Any:
    """Lazily load a neural detector from controlnet_aux.

    Raises:
        ImportError: If ``controlnet-aux`` is not installed.
    """
    global _AUX_DETECTOR_REGISTRY  # noqa: PLW0603

    if detector_type in _AUX_DETECTOR_REGISTRY:
        return _AUX_DETECTOR_REGISTRY[detector_type]

    try:
        from controlnet_aux import HEDdetector, LineartDetector, PidiNetDetector
    except ImportError as exc:
        raise ImportError(
            f"controlnet-aux is required for mode '{detector_type}'. "
            "Install it with: pip install controlnet-aux>=0.0.7"
        ) from exc

    cls_map = {
        "hed": HEDdetector,
        "softedge": PidiNetDetector,
        "pidinet": PidiNetDetector,
        "lineart": LineartDetector,
        "lineart_coarse": LineartDetector,
    }

    if detector_type not in cls_map:
        raise ValueError(f"Unknown neural detector: {detector_type}")

    logger.info("Loading '%s' detector weights from Hugging Face...", detector_type)
    detector = cls_map[detector_type].from_pretrained("lllyasviel/Annotators")
    _AUX_DETECTOR_REGISTRY[detector_type] = detector
    return detector


# ---------------------------------------------------------------------------
# 5. Unified Conditioning Entry Point
# ---------------------------------------------------------------------------


def prepare_condition(
    image: str | Path | Image.Image | np.ndarray,
    mode: str = "clean",
    resolution: int | tuple[int, int] = 512,
    multiple: int = 64,
    **kwargs: Any,
) -> Image.Image:
    """Convert any hand-drawn sketch into a canonical ControlNet conditioning map.

    Args:
        image: Input sketch (path, PIL Image, or numpy array).
        mode: Conditioning extraction mode:
            - ``'clean'`` / ``'sketch'``: Custom cleaned binary sketch (white on black).
            - ``'canny'`` : Canny edge detection.
            - ``'hed'``   : HED detector (via controlnet_aux).
            - ``'pidinet'`` / ``'softedge'`` : PiDiNet soft-edge detector.
            - ``'lineart'``: Neural lineart extractor.
            - ``'none'``  : Raw sketch oriented, padded, and converted to RGB.
        resolution: Target size (int for square or (w, h)).
        multiple: Required dimension multiple (default 64).
        **kwargs: Mode-specific parameters:
            - ``adaptive_block_size``, ``adaptive_c``, ``min_blob_area``, ``line_thickness`` (for 'clean')
            - ``low_threshold``, ``high_threshold`` (for 'canny')
            - ``coarse`` (for 'lineart')

    Returns:
        RGB PIL Image with dimensions divisible by ``multiple``, ready for ControlNet.
    """
    mode = mode.lower().strip()

    # Step 1: Cleaned or oriented base
    if mode in ("clean", "sketch", "raw_clean"):
        cleaned_arr = clean_sketch(
            image,
            adaptive_block_size=kwargs.get("adaptive_block_size", 25),
            adaptive_c=kwargs.get("adaptive_c", 10),
            min_blob_area=kwargs.get("min_blob_area", 20),
            line_thickness=kwargs.get("line_thickness", 0),
        )
        padded = resize_and_pad(cleaned_arr, resolution=resolution, multiple=multiple)
        # Convert to 3-channel RGB for ControlNet
        rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    if mode == "canny":
        gray = load_and_orient_image(image)
        edges = extract_canny(
            gray,
            low_threshold=kwargs.get("low_threshold", 100),
            high_threshold=kwargs.get("high_threshold", 200),
        )
        padded = resize_and_pad(edges, resolution=resolution, multiple=multiple)
        rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    if mode in ("hed", "pidinet", "softedge", "lineart", "lineart_coarse"):
        # Load and orient to PIL RGB first
        if isinstance(image, (str, Path)):
            pil_input = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_input = image
        else:
            pil_input = Image.fromarray(image)

        pil_input = ImageOps.exif_transpose(pil_input).convert("RGB")
        try:
            detector = _get_aux_detector(mode)
        except (ImportError, Exception) as exc:
            logger.warning("Neural detector '%s' unavailable (%s); falling back to 'clean' mode.", mode, exc)
            return prepare_condition(image, mode="clean", resolution=resolution, multiple=multiple, **kwargs)

        res_val = resolution if isinstance(resolution, int) else max(resolution)
        det_kwargs = {"detect_resolution": res_val, "image_resolution": res_val}
        if mode == "lineart_coarse" or kwargs.get("coarse"):
            det_kwargs["coarse"] = True

        detected = detector(pil_input, **det_kwargs)
        # Pad to multiple of 64
        padded = resize_and_pad(np.array(detected), resolution=resolution, multiple=multiple)
        if padded.ndim == 2:
            padded = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(padded)

    if mode == "none":
        gray = load_and_orient_image(image)
        padded = resize_and_pad(gray, resolution=resolution, multiple=multiple)
        rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    raise ValueError(
        f"Unknown mode '{mode}'. Choose from: clean, canny, hed, pidinet, softedge, lineart, none."
    )


# ---------------------------------------------------------------------------
# 6. Backward Compatibility: SketchPreprocessor & load_sketch
# ---------------------------------------------------------------------------


class SketchPreprocessor:
    """Wrapper class providing backward compatibility for diffusers pipelines.

    Args:
        preprocessor_type: Conditioning mode ('clean', 'hed', 'pidinet', 'lineart', 'canny', 'none').
        resolution: Target image resolution.
    """

    def __init__(self, preprocessor_type: str = "clean", resolution: int = 512) -> None:
        self.preprocessor_type = preprocessor_type
        self.resolution = resolution

    def __call__(self, image: Image.Image) -> Image.Image:
        """Process image into conditioning map."""
        return prepare_condition(image, mode=self.preprocessor_type, resolution=self.resolution)


def load_sketch(path: str | Path, resolution: int | None = None) -> Image.Image:
    """Load a sketch and optionally resize it preserving aspect ratio with padding.

    Args:
        path: Path to the sketch.
        resolution: Target square resolution (multiple of 64).

    Returns:
        RGB PIL Image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sketch not found: {path}")

    pil_img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if resolution is not None:
        arr = resize_and_pad(np.array(pil_img), resolution=resolution, multiple=64)
        return Image.fromarray(arr)
    return pil_img


def _resize_keep_aspect(image: Image.Image, max_side: int) -> Image.Image:
    """Resize keeping aspect ratio (legacy helper for backward compatibility)."""
    w, h = image.size
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    return image.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# 7. Comparison Grid Helper & CLI Interface
# ---------------------------------------------------------------------------


def create_conditioning_grid(
    original_sketch: Image.Image,
    conditions: dict[str, Image.Image],
    target_height: int = 256,
) -> Image.Image:
    """Create a side-by-side comparison banner: Original -> each condition mode.

    Args:
        original_sketch: Input original sketch.
        conditions: Dict mapping mode name -> condition PIL image.
        target_height: Height of each panel in the banner.

    Returns:
        Combined PIL Image grid.
    """
    panels: list[tuple[str, Image.Image]] = [("Original", original_sketch.convert("RGB"))]
    for name, img in conditions.items():
        panels.append((name.capitalize(), img.convert("RGB")))

    # Resize all panels to same height
    resized_panels = []
    total_width = 0
    font_height = 30

    for title, img in panels:
        w, h = img.size
        new_w = int(w * (target_height / h))
        panel_img = img.resize((new_w, target_height), Image.LANCZOS)
        resized_panels.append((title, panel_img))
        total_width += new_w

    # Create banner canvas with header space for labels
    canvas_h = target_height + font_height
    grid = Image.new("RGB", (total_width, canvas_h), color=(20, 20, 20))

    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    x_offset = 0
    for title, panel_img in resized_panels:
        grid.paste(panel_img, (x_offset, font_height))
        draw.text((x_offset + 10, 6), title, fill=(240, 240, 240), font=font)
        x_offset += panel_img.width

    return grid


def main() -> None:
    """CLI runner: process sketch directory or image into conditioning maps."""
    parser = argparse.ArgumentParser(
        description="Preprocess hand-drawn sketches into ControlNet conditioning maps."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="sketches",
        help="Input directory of sketches or path to a single sketch file.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="outputs/conditions",
        help="Directory to save generated conditioning maps.",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=512,
        help="Target output resolution (default 512).",
    )
    parser.add_argument(
        "--modes",
        "-m",
        default="clean,canny",
        help="Comma-separated modes: clean, canny, hed, pidinet, lineart, none.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # Collect files
    if input_path.is_file():
        sketch_files = [input_path]
    elif input_path.is_dir():
        sketch_files = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
        )
    else:
        print(f"Error: Input path '{input_path}' not found.")
        return

    if not sketch_files:
        print(f"No image files found in '{input_path}'.")
        return

    print(f"Found {len(sketch_files)} sketch(es) in '{input_path}'.")
    print(f"Processing modes: {modes} at resolution {args.resolution}x{args.resolution}...\n")

    for sketch_path in sketch_files:
        print(f"Processing: {sketch_path.name}")
        orig_img = ImageOps.exif_transpose(Image.open(sketch_path)).convert("RGB")
        processed_conditions: dict[str, Image.Image] = {}

        for mode in modes:
            try:
                cond = prepare_condition(sketch_path, mode=mode, resolution=args.resolution)
                out_file = output_dir / f"{sketch_path.stem}_{mode}.png"
                cond.save(out_file)
                processed_conditions[mode] = cond
                print(f"  ✓ {mode:<10} -> {out_file}")
            except Exception as exc:
                print(f"  ✗ {mode:<10} failed: {exc}")

        # Save side-by-side comparison grid if we have conditions
        if processed_conditions:
            grid_img = create_conditioning_grid(orig_img, processed_conditions)
            grid_file = output_dir / f"{sketch_path.stem}_grid.png"
            grid_img.save(grid_file)
            print(f"  ★ Comparison grid saved: {grid_file}\n")


if __name__ == "__main__":
    main()
