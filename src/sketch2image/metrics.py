"""Objective metrics for sketch preservation, prompt alignment and compute cost.

Structure preservation
    Edges are extracted from the *generated* image (Canny, HED or PiDiNet) and
    compared with the *cleaned sketch* placed in exactly the same frame the
    generator saw (``prepare_condition(sketch, "clean", resolution)``), so the
    padding / aspect handling of the pipeline never shows up as an error.

    Both maps are thinned to 1-px skeletons and compared at a fixed evaluation
    resolution (512 px) so SD 1.5 (512) and SDXL (1024) outputs are scored on
    the same pixel scale.

    * :func:`edge_f1_iou` - precision / recall / F1 with a distance tolerance
      ``tau`` (a generated edge pixel counts as correct if a sketch pixel lies
      within ``tau`` px, and vice versa), plus a tolerant IoU of the two maps
      dilated by ``tau``.
    * :func:`chamfer_distance` - mean bidirectional nearest-edge distance (px).
      Reported per direction too: sketch->generated measures *missing* lines,
      generated->sketch measures *hallucinated* lines. When the image renders
      the strokes as dark lines, Canny finds both stroke borders, so Chamfer has
      a floor of ~half the stroke width (hence default ``tau`` = 4 px).
    * :func:`edge_ssim` - SSIM between the two edge maps after a Gaussian blur
      of sigma ~ tau (raw binary-map SSIM is dominated by empty background and
      punishes 1-px shifts; blurring makes it a soft, shift-tolerant overlap).

    Interpretation: generated images legitimately contain texture edges the
    sketch does not (foliage, reflections), so *precision* is expected to be
    low-ish and *recall* is the most direct "did the sketch survive" number.

Prompt alignment
    :class:`ClipScorer` - CLIPScore (Hessel et al., 2021) in the torchmetrics
    convention ``100 * max(cos(img, text), 0)`` using Hugging Face
    ``openai/clip-vit-base-patch32`` (MIT licence). ViT-B/32 is deliberately
    *not* the ViT-L/14 text encoder that SD 1.5 is conditioned on, which would
    bias the score toward the generator. There is no heuristic fallback: if
    ``transformers`` is missing an error is raised rather than a fake number.

Image-quality proxy (LAION aesthetic predictor) - intentionally skipped
    It needs CLIP ViT-L/14 (~1.7 GB) plus MLP weights that are distributed via
    GitHub rather than a verified Hugging Face model ID (project policy: no
    unverified IDs), it was trained on LAION photographs and is unreliable on
    the watercolour / concept-art presets we compare, and image quality is not
    something our research questions (sketch fidelity vs. prompt adherence)
    depend on. Visual inspection of the grids covers quality instead.

Runtime / peak VRAM
    Recorded at generation time in each ``*_meta.json`` sidecar
    (see :meth:`sketch2image.pipelines.SketchToImageGenerator.generate`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from sketch2image.preprocessing import extract_canny, prepare_condition

logger = logging.getLogger(__name__)

EVAL_RESOLUTION = 512
DEFAULT_TAU = 4
EDGE_METHODS = ("canny", "hed", "pidinet")
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

ImageLike = str | Path | Image.Image | np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_rgb(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def _to_binary(edge_map: np.ndarray, threshold: int = 127) -> np.ndarray:
    if edge_map.ndim == 3:
        edge_map = cv2.cvtColor(edge_map, cv2.COLOR_RGB2GRAY)
    return (edge_map > threshold).astype(np.uint8)


def _resize_binary(binary: np.ndarray, size: int) -> np.ndarray:
    if binary.shape == (size, size):
        return binary
    # Area-resize then re-threshold keeps thin lines from vanishing.
    resized = cv2.resize(
        binary.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA
    )
    return (resized > 0.0).astype(np.uint8)


def thin(binary: np.ndarray) -> np.ndarray:
    """Reduce a binary stroke map to 1-px wide centre lines.

    Args:
        binary: 2D array, non-zero = stroke.

    Returns:
        2D uint8 array in {0, 1}.
    """
    from skimage.morphology import skeletonize

    return skeletonize(binary > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Edge maps
# ---------------------------------------------------------------------------


def reference_edges(
    sketch: ImageLike,
    resolution: int,
    eval_resolution: int = EVAL_RESOLUTION,
) -> np.ndarray:
    """Cleaned sketch in the generator's frame, thinned, at evaluation size.

    Args:
        sketch: Raw input sketch (path / image / array).
        resolution: Resolution the generator used (512 SD 1.5, 1024 SDXL); the
            same resize-and-pad is applied so coordinates line up.
        eval_resolution: Square size all comparisons are done at.

    Returns:
        2D uint8 skeleton in {0, 1}, shape ``(eval_resolution, eval_resolution)``.
    """
    cond = np.array(prepare_condition(sketch, mode="clean", resolution=resolution))
    return thin(_resize_binary(_to_binary(cond), eval_resolution))


def generated_edges(
    image: ImageLike,
    method: str = "canny",
    eval_resolution: int = EVAL_RESOLUTION,
    canny_low: int = 100,
    canny_high: int = 200,
    soft_threshold: int = 96,
) -> np.ndarray:
    """Extract a thinned edge map from a generated image.

    Args:
        image: Generated RGB image.
        method: ``"canny"`` (classical, no weights), ``"hed"`` or ``"pidinet"``
            (neural, via ``controlnet_aux`` / ``lllyasviel/Annotators``; these
            need the GPU environment's dependencies and download weights).
        eval_resolution: Square size the image is resized to before detection.
        canny_low: Canny lower hysteresis threshold.
        canny_high: Canny upper hysteresis threshold.
        soft_threshold: Binarisation threshold (0-255) for HED/PiDiNet soft maps.

    Returns:
        2D uint8 skeleton in {0, 1}.

    Raises:
        ValueError: Unknown ``method``.
        ImportError: ``controlnet_aux`` missing for a neural method.
    """
    rgb = _to_rgb(image).resize(
        (eval_resolution, eval_resolution), Image.Resampling.LANCZOS
    )
    method = method.lower()
    if method == "canny":
        return thin(_to_binary(extract_canny(rgb, canny_low, canny_high)))
    if method in ("hed", "pidinet"):
        from sketch2image.preprocessing import _get_aux_detector

        detector = _get_aux_detector(method)
        soft = detector(
            rgb, detect_resolution=eval_resolution, image_resolution=eval_resolution
        )
        soft_arr = np.array(
            soft.convert("L").resize((eval_resolution, eval_resolution))
        )
        return thin(_to_binary(soft_arr, soft_threshold))
    raise ValueError(f"Unknown edge method '{method}'. Choose from {EDGE_METHODS}.")


# ---------------------------------------------------------------------------
# Structure metrics (all take two {0,1} maps of equal shape)
# ---------------------------------------------------------------------------


def _dist_to(binary: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest non-zero pixel."""
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(binary == 0)


def edge_f1_iou(
    ref: np.ndarray, gen: np.ndarray, tau: float = DEFAULT_TAU
) -> dict[str, float]:
    """Tolerant edge precision, recall, F1 and IoU.

    Args:
        ref: Reference (sketch) edge map, {0,1}.
        gen: Generated-image edge map, {0,1}, same shape.
        tau: Distance tolerance in pixels.

    Returns:
        ``precision``, ``recall``, ``f1``, ``iou`` and ``density``
        (generated edge pixels / sketch edge pixels; >1 means extra detail).
    """
    n_ref, n_gen = int(ref.sum()), int(gen.sum())
    if n_ref == 0 or n_gen == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "density": n_gen / max(n_ref, 1),
        }

    precision = float(((_dist_to(ref) <= tau) & (gen > 0)).sum() / n_gen)
    recall = float(((_dist_to(gen) <= tau) & (ref > 0)).sum() / n_ref)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    size = 2 * int(np.ceil(tau)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    ref_d, gen_d = cv2.dilate(ref, kernel) > 0, cv2.dilate(gen, kernel) > 0
    iou = float((ref_d & gen_d).sum() / max((ref_d | gen_d).sum(), 1))

    return {
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
        "iou": iou,
        "density": n_gen / n_ref,
    }


def chamfer_distance(ref: np.ndarray, gen: np.ndarray) -> dict[str, float]:
    """Bidirectional Chamfer distance between two edge maps (pixels, lower = better).

    Args:
        ref: Reference (sketch) edge map, {0,1}.
        gen: Generated-image edge map, {0,1}, same shape.

    Returns:
        ``chamfer`` (mean of both directions), ``chamfer_s2g`` (sketch->gen,
        large = sketch lines missing), ``chamfer_g2s`` (gen->sketch, large =
        extra structure). Empty maps give the image diagonal.
    """
    if ref.sum() == 0 or gen.sum() == 0:
        diag = float(np.hypot(*ref.shape))
        return {"chamfer": diag, "chamfer_s2g": diag, "chamfer_g2s": diag}
    s2g = float(_dist_to(gen)[ref > 0].mean())
    g2s = float(_dist_to(ref)[gen > 0].mean())
    return {"chamfer": (s2g + g2s) / 2, "chamfer_s2g": s2g, "chamfer_g2s": g2s}


def edge_ssim(
    ref: np.ndarray, gen: np.ndarray, sigma: float = DEFAULT_TAU / 2
) -> float:
    """SSIM between Gaussian-blurred edge maps.

    Args:
        ref: Reference edge map, {0,1}.
        gen: Generated edge map, {0,1}, same shape.
        sigma: Blur applied to both maps before SSIM (shift tolerance).

    Returns:
        SSIM in [-1, 1].
    """
    from skimage.metrics import structural_similarity

    a = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), sigma)
    b = cv2.GaussianBlur(gen.astype(np.float32), (0, 0), sigma)
    hi = max(float(a.max()), float(b.max()), 1e-6)
    return float(structural_similarity(a / hi, b / hi, data_range=1.0))


def structure_metrics(
    ref: np.ndarray, gen: np.ndarray, tau: float = DEFAULT_TAU
) -> dict[str, float]:
    """All structure metrics for one (reference, generated) edge-map pair.

    Args:
        ref: Reference (sketch) edge map, {0,1}.
        gen: Generated-image edge map, {0,1}, same shape.
        tau: Distance tolerance in pixels.

    Returns:
        Merged dict of :func:`edge_f1_iou`, :func:`chamfer_distance` and ``ssim``.
    """
    if ref.shape != gen.shape:
        raise ValueError(f"Edge maps differ in shape: {ref.shape} vs {gen.shape}")
    out = edge_f1_iou(ref, gen, tau)
    out.update(chamfer_distance(ref, gen))
    out["ssim"] = edge_ssim(ref, gen, sigma=max(tau / 2, 1.0))
    return out


# ---------------------------------------------------------------------------
# Prompt alignment
# ---------------------------------------------------------------------------


class ClipScorer:
    """Lazily-loaded CLIPScore model (Hugging Face ``transformers``).

    Args:
        model_id: Verified HF model ID.
        device: ``"cuda"``/``"cpu"``; auto-detected when ``None``.
    """

    def __init__(
        self, model_id: str = DEFAULT_CLIP_MODEL, device: str | None = None
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CLIP '%s' on %s", self.model_id, self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()

    def score(self, image: ImageLike, text: str) -> float:
        """CLIPScore = 100 * max(cosine(image, text), 0).

        Args:
            image: Generated image.
            text: Prompt (truncated to CLIP's 77-token context).

        Returns:
            Score, typically 20-35 for matching image/prompt pairs.
        """
        import torch

        if self._model is None:
            self._load()
        inputs = self._processor(
            text=[text],
            images=_to_rgb(image),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float(100.0 * torch.clamp((img * txt).sum(), min=0.0))


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------


@dataclass
class MetricConfig:
    """Settings for :func:`evaluate_generation`.

    Attributes:
        edge_methods: Edge extractors applied to the generated image.
        tau: Edge-matching tolerance (px at ``eval_resolution``).
        eval_resolution: Common comparison size.
    """

    edge_methods: tuple[str, ...] = EDGE_METHODS
    tau: float = DEFAULT_TAU
    eval_resolution: int = EVAL_RESOLUTION


def evaluate_generation(
    sketch: ImageLike,
    generated: ImageLike,
    resolution: int,
    prompts: dict[str, str] | None = None,
    clip: ClipScorer | None = None,
    config: MetricConfig | None = None,
    ref: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the metric suite for one generated image.

    Args:
        sketch: Raw input sketch.
        generated: Generated image.
        resolution: Generation resolution (to reproduce the condition frame).
        prompts: ``{column_suffix: prompt}`` to CLIP-score against, e.g.
            ``{"": raw_prompt, "_full": styled_prompt}`` -> ``clip_score``,
            ``clip_score_full``. Ignored when ``clip`` is ``None``.
        clip: CLIP scorer; ``None`` skips prompt alignment.
        config: Metric settings.
        ref: Pre-computed :func:`reference_edges` (saves work across a sweep).

    Returns:
        Flat dict, structure keys prefixed by edge method, e.g. ``hed_f1``,
        ``canny_chamfer``.
    """
    cfg = config or MetricConfig()
    if ref is None:
        ref = reference_edges(sketch, resolution, cfg.eval_resolution)
    gen_img = _to_rgb(generated)

    out: dict[str, float] = {}
    for method in cfg.edge_methods:
        gen = generated_edges(gen_img, method, cfg.eval_resolution)
        for key, val in structure_metrics(ref, gen, cfg.tau).items():
            out[f"{method}_{key}"] = val

    if clip is not None:
        for suffix, text in (prompts or {}).items():
            out[f"clip_score{suffix}"] = clip.score(gen_img, text)
    return out
