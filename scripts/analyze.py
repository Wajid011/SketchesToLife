#!/usr/bin/env python3
"""Compute objective metrics for every experiment output and write the analysis.

Reads every ``*_meta.json`` sidecar under ``--outputs-dir`` (written by
``scripts/run_experiments.py``), scores each generated image with
:mod:`sketch2image.metrics`, and writes to ``--results-dir``:

* ``metrics.csv``                - one row per generated image
* ``scale_sweep.png``            - edge F1 (per extractor) and CLIP vs conditioning scale (E2)
* ``backend_comparison.png``     - edge F1 / CLIP / runtime / peak VRAM per backend (E1)
* ``tradeoff_scatter.png``       - sketch fidelity vs prompt alignment, all images
* ``ANALYSIS.md``                - plain-language findings generated from the numbers

Typical use
    Colab (full suite; downloads HED/PiDiNet annotators + CLIP ViT-B/32)::

        python scripts/analyze.py

    Local CPU laptop (no model downloads - Canny only, no CLIP)::

        python scripts/analyze.py --edge-methods canny --no-clip

Outputs produced by the CPU *mock* generator (``"mock": true`` in the sidecar)
are excluded when real outputs exist; if only mock outputs exist they are kept
so the pipeline can be smoke-tested, and ANALYSIS.md is stamped as invalid.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from sketch2image.metrics import (  # noqa: E402
    DEFAULT_CLIP_MODEL,
    DEFAULT_TAU,
    EDGE_METHODS,
    EVAL_RESOLUTION,
    ClipScorer,
    MetricConfig,
    evaluate_generation,
    reference_edges,
)

logger = logging.getLogger("analyze")

# Validated categorical palette (dataviz reference, light mode), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, SURFACE = "#2b2b2a", "#6b6a66", "#e4e3de", "#fcfcfb"

# Fixed backend order -> fixed colour (colour follows the entity, never the rank).
BACKEND_ORDER = [
    "SD1.5 scribble",
    "SD1.5 lineart",
    "SD1.5 softedge",
    "SDXL scribble",
    "T2I-Adapter sketch",
]
BACKEND_COLOR = dict(zip(BACKEND_ORDER, SERIES, strict=False))

CONFIG_COLS = [
    "backend_label",
    "condition_mode",
    "preset",
    "conditioning_scale",
    "guidance_scale",
    "guidance_end",
    "steps",
    "scheduler",
    "resolution",
]


# ---------------------------------------------------------------------------
# Collect + score
# ---------------------------------------------------------------------------


def backend_label(backend: str, cn_type: str | None) -> str:
    """Human-readable backend name used in tables and plots."""
    if backend == "sd15_controlnet":
        return f"SD1.5 {cn_type or 'scribble'}"
    if backend == "sdxl_controlnet":
        return f"SDXL {cn_type or 'scribble'}"
    if backend == "t2i_adapter":
        return "T2I-Adapter sketch"
    return backend


def _resolve_sketch(
    meta: dict[str, Any], sketches_dir: Path, meta_path: Path
) -> Path | None:
    raw = meta.get("sketch_file") or ""
    if raw:
        p = Path(raw)
        if p.exists():
            return p
        name = PureWindowsPath(raw).name if "\\" in raw else Path(raw).name
        if (sketches_dir / name).exists():
            return sketches_dir / name
    for cand in sorted(sketches_dir.glob("*")):
        if cand.suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg",
        ) and meta_path.name.startswith(cand.stem):
            return cand
    return None


def find_records(outputs_dir: Path, sketches_dir: Path) -> list[dict[str, Any]]:
    """One record per generated image listed in a ``*_meta.json`` sidecar."""
    records = []
    for meta_path in sorted(outputs_dir.glob("**/*_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sketch = _resolve_sketch(meta, sketches_dir, meta_path)
        if sketch is None:
            logger.warning("No sketch found for %s - skipped", meta_path.name)
            continue
        files = meta.get("generated_files") or []
        for idx, fname in enumerate(files):
            img = meta_path.parent / fname
            if img.exists():
                records.append(
                    {
                        "meta": meta,
                        "meta_path": meta_path,
                        "image": img,
                        "image_index": idx,
                        "sketch": sketch,
                    }
                )
    return records


def score_records(
    records: list[dict[str, Any]], cfg: MetricConfig, clip: ClipScorer | None
) -> pd.DataFrame:
    """Run the metric suite on every record."""
    ref_cache: dict[tuple[str, int], np.ndarray] = {}
    rows = []
    for i, rec in enumerate(records, 1):
        m = rec["meta"]
        res = int(m.get("resolution", 512))
        key = (str(rec["sketch"]), res)
        if key not in ref_cache:
            ref_cache[key] = reference_edges(rec["sketch"], res, cfg.eval_resolution)
        raw_prompt = m.get("raw_prompt") or m.get("prompt", "")
        full_prompt = m.get("prompt", raw_prompt)
        metrics = evaluate_generation(
            rec["sketch"],
            rec["image"],
            res,
            prompts={"": raw_prompt, "_full": full_prompt},
            clip=clip,
            config=cfg,
            ref=ref_cache[key],
        )
        rows.append(
            {
                "experiment": rec["meta_path"]
                .parents[1]
                .name,  # outputs/experiments/<E>/<sketch>/
                "sketch": rec["sketch"].stem,
                "backend": m.get("backend"),
                "controlnet_type": m.get("controlnet_type"),
                "backend_label": backend_label(
                    m.get("backend", ""), m.get("controlnet_type")
                ),
                "condition_mode": m.get("condition_mode", "clean"),
                "preset": m.get("style_preset") or "none",
                "conditioning_scale": float(
                    m.get("controlnet_conditioning_scale", 1.0)
                ),
                "guidance_scale": float(m.get("guidance_scale", 7.5)),
                "guidance_end": float(m.get("control_guidance_end", 1.0)),
                "steps": int(m.get("num_inference_steps", 30)),
                "scheduler": m.get("scheduler", ""),
                "seed": int(m.get("seed", 42)),
                "resolution": res,
                **metrics,
                "runtime_sec": float(m.get("runtime_sec", np.nan)),
                "peak_vram_mb": float(m.get("peak_vram_mb", np.nan)),
                "mock": bool(m.get("mock", False)),
                "prompt": raw_prompt,
                "image_path": rec["image"].as_posix(),
            }
        )
        if i % 10 == 0 or i == len(records):
            logger.info("Scored %d/%d", i, len(records))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _fig(ncols: int, w: float, h: float) -> tuple[plt.Figure, Any]:
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_scale_sweep(
    e2: pd.DataFrame, methods: list[str], has_clip: bool, path: Path
) -> None:
    """Edge F1 (one line per extractor) and CLIP vs conditioning scale."""
    fig, axes = _fig(2, 10, 3.8)
    g = e2.groupby("conditioning_scale")
    ax = axes[0]
    for i, m in enumerate(methods):
        mean = g[f"{m}_f1"].mean()
        ax.plot(
            mean.index,
            mean.values,
            color=SERIES[i],
            lw=2,
            marker="o",
            ms=6,
            label=m.upper(),
        )
        ax.annotate(
            m.upper(),
            (mean.index[-1], mean.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    ax.scatter(
        e2["conditioning_scale"],
        e2[f"{methods[0]}_f1"],
        s=12,
        color=MUTED,
        alpha=0.45,
        zorder=1,
        label=f"per sketch ({methods[0].upper()})",
    )
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(
        ax,
        "Sketch preservation",
        "ControlNet conditioning scale",
        f"Edge F1 (τ = {DEFAULT_TAU:g} px)",
    )

    ax = axes[1]
    if has_clip:
        mean = g["clip_score"].mean()
        ax.scatter(
            e2["conditioning_scale"],
            e2["clip_score"],
            s=12,
            color=MUTED,
            alpha=0.45,
            label="per sketch",
        )
        ax.plot(
            mean.index,
            mean.values,
            color=SERIES[0],
            lw=2,
            marker="o",
            ms=6,
            label="mean",
        )
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "CLIP not computed (--no-clip)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=MUTED,
        )
    _style(ax, "Prompt alignment", "ControlNet conditioning scale", "CLIPScore")
    fig.suptitle(
        "E2 · SD1.5 scribble: conditioning-scale sweep",
        x=0.01,
        ha="left",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
    )
    _save(fig, path)


def plot_backend_comparison(
    e1: pd.DataFrame, primary: str, has_clip: bool, path: Path
) -> None:
    """Small multiples: edge F1, CLIP, runtime, peak VRAM per backend."""
    agg = e1.groupby("backend_label").agg(
        f1=(f"{primary}_f1", "mean"),
        clip=("clip_score", "mean"),
        runtime=("runtime_sec", "mean"),
        vram=("peak_vram_mb", "max"),
    )
    order = [b for b in BACKEND_ORDER if b in agg.index] + [
        b for b in agg.index if b not in BACKEND_ORDER
    ]
    agg = agg.loc[order[::-1]]
    colors = [BACKEND_COLOR.get(b, MUTED) for b in agg.index]
    panels = [
        ("f1", f"Edge F1 ({primary.upper()})", "{:.2f}"),
        ("clip", "CLIPScore", "{:.1f}"),
        ("runtime", "Runtime / image (s)", "{:.1f}"),
        ("vram", "Peak VRAM (GB)", "{:.1f}"),
    ]
    if not has_clip:
        panels.pop(1)
    fig, axes = _fig(len(panels), 3.1 * len(panels), 0.55 * len(agg) + 1.4)
    for k, (ax, (col, title, fmt)) in enumerate(zip(axes, panels, strict=False)):
        vals = agg[col] / 1024 if col == "vram" else agg[col]
        ax.barh(
            range(len(agg)),
            vals,
            color=colors,
            height=0.62,
            edgecolor=SURFACE,
            linewidth=2,
        )
        for y, v in enumerate(vals):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(agg)))
        ax.set_yticklabels(agg.index if k == 0 else [""] * len(agg))
        ax.set_xlim(0, vals.max() * 1.25 if vals.max() > 0 else 1.0)
        _style(ax, title)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "E1 · backend comparison (same sketches, prompts, seed)",
        x=0.01,
        ha="left",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
    )
    _save(fig, path)


def plot_tradeoff(df: pd.DataFrame, primary: str, path: Path) -> None:
    """Fidelity (edge F1) vs alignment (CLIP) for every image."""
    fig, ax = _fig(1, 7, 4.8)
    for b in [b for b in BACKEND_ORDER if b in set(df["backend_label"])]:
        sub = df[df["backend_label"] == b]
        ax.scatter(
            sub["clip_score"],
            sub[f"{primary}_f1"],
            s=46,
            color=BACKEND_COLOR[b],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=b,
            alpha=0.9,
        )
    e2 = df[df["experiment"] == "E2"]
    if not e2.empty:
        path_pts = e2.groupby("conditioning_scale")[
            ["clip_score", f"{primary}_f1"]
        ].mean()
        ax.plot(
            path_pts["clip_score"],
            path_pts[f"{primary}_f1"],
            color=INK,
            lw=1.2,
            ls="--",
            zorder=0,
            label="E2 mean, by scale",
        )
        for s, r in path_pts.iterrows():
            ax.annotate(
                f"{s:g}",
                (r["clip_score"], r[f"{primary}_f1"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
    ax.legend(frameon=False, fontsize=8, loc="best")
    _style(
        ax,
        "Sketch fidelity vs prompt alignment (all images)",
        "CLIPScore  →  more on-prompt",
        f"Edge F1, {primary.upper()}  →  more sketch-faithful",
    )
    _save(fig, path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    fmt = fmt or {}
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(map(str, cols)) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("–")
            elif c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rel(p: str, results_dir: Path) -> str:
    try:
        return Path(
            os.path.relpath(Path(p).resolve(), results_dir.resolve())
        ).as_posix()
    except ValueError:  # different drive on Windows
        return p


def summarise(df: pd.DataFrame, primary: str, has_clip: bool) -> dict[str, Any]:
    """Compute every number the report and console summary quote."""
    P = primary
    s: dict[str, Any] = {
        "n": len(df),
        "sketches": sorted(df["sketch"].unique()),
        "mock": bool(df["mock"].any()),
    }
    cols = {
        "F1": f"{P}_f1",
        "Recall": f"{P}_recall",
        "Precision": f"{P}_precision",
        "IoU": f"{P}_iou",
        "Chamfer px": f"{P}_chamfer",
        "Edge SSIM": f"{P}_ssim",
    }
    if has_clip:
        cols["CLIP"] = "clip_score"
    cols.update({"Runtime s": "runtime_sec", "Peak VRAM MB": "peak_vram_mb"})

    e1 = df[df["experiment"] == "E1"]
    if not e1.empty:
        t = e1.groupby("backend_label")[list(cols.values())].mean()
        t["peak_vram_mb"] = e1.groupby("backend_label")["peak_vram_mb"].max()
        t.columns = list(cols.keys())
        order = [b for b in BACKEND_ORDER if b in t.index]
        s["e1"] = (
            t.loc[order].reset_index().rename(columns={"backend_label": "Backend"})
        )

    e2 = df[df["experiment"] == "E2"]
    if not e2.empty:
        sc = e2.groupby("conditioning_scale")[
            [
                f"{P}_f1",
                f"{P}_recall",
                f"{P}_precision",
                f"{P}_chamfer_s2g",
                f"{P}_chamfer_g2s",
                f"{P}_density",
            ]
            + (["clip_score"] if has_clip else [])
        ].mean()
        s["e2"] = sc
        rec = sc[f"{P}_recall"]
        s["scale_struct_ok"] = float(rec[rec >= 0.9 * rec.max()].index.min())
        s["scale_best_f1"] = float(sc[f"{P}_f1"].idxmax())
        if has_clip:
            s["scale_best_clip"] = float(sc["clip_score"].idxmax())
            s["corr_scale_clip"] = float(
                e2[["conditioning_scale", "clip_score"]].corr("spearman").iloc[0, 1]
            )
        s["corr_scale_f1"] = float(
            e2[["conditioning_scale", f"{P}_f1"]].corr("spearman").iloc[0, 1]
        )
        per_sketch = e2.pivot_table(
            index="sketch", columns="conditioning_scale", values=f"{P}_recall"
        )
        s["e2_per_sketch"] = per_sketch

    if has_clip and len(df) > 2:
        s["corr_f1_clip"] = float(
            df[[f"{P}_f1", "clip_score"]].corr("spearman").iloc[0, 1]
        )

    # Best setting per backend: most on-prompt config whose F1 is within 90 % of
    # that backend's best F1 (= "keeps the sketch, then maximise the prompt").
    best = []
    for b, sub in df.groupby("backend_label"):
        g = (
            sub.groupby(CONFIG_COLS[1:], dropna=False)
            .agg(
                f1=(f"{P}_f1", "mean"),
                n=(f"{P}_f1", "size"),
                clip=("clip_score", "mean") if has_clip else (f"{P}_f1", "mean"),
            )
            .reset_index()
        )
        ok = g[g["f1"] >= 0.9 * g["f1"].max()]
        pick = ok.loc[ok["clip"].idxmax()] if has_clip else g.loc[g["f1"].idxmax()]
        varied = [c for c in CONFIG_COLS[1:] if g[c].nunique() > 1]
        setting = (
            ", ".join(f"{c}={pick[c]}" for c in varied)
            or "default (only setting tested)"
        )
        best.append(
            {
                "Backend": b,
                "Configs tried": len(g),
                "Recommended": setting,
                "F1": pick["f1"],
                **({"CLIP": pick["clip"]} if has_clip else {}),
            }
        )
    order = {b: i for i, b in enumerate(BACKEND_ORDER)}
    s["best"] = pd.DataFrame(best).sort_values("Backend", key=lambda c: c.map(order))
    return s


def failure_cases(
    df: pd.DataFrame, primary: str, has_clip: bool, k: int = 3
) -> dict[str, pd.DataFrame]:
    """Pick the images most representative of each failure mode."""
    P = primary
    show = (
        [
            "experiment",
            "sketch",
            "backend_label",
            "conditioning_scale",
            f"{P}_recall",
            f"{P}_precision",
            f"{P}_chamfer_s2g",
            f"{P}_chamfer_g2s",
        ]
        + (["clip_score"] if has_clip else [])
        + ["image_path"]
    )
    out = {
        "ignored": df.nsmallest(k, f"{P}_recall")[show],
        "hallucinated": df.nlargest(k, f"{P}_chamfer_g2s")[show],
    }
    if has_clip:
        # Over-constrained: high scale AND below that sketch's median CLIP.
        med = df.groupby("sketch")["clip_score"].transform("median")
        hi = df[(df["conditioning_scale"] >= 1.2) & (df["clip_score"] < med)]
        out["over"] = hi.nsmallest(k, "clip_score")[show]
    return out


def write_report(
    df: pd.DataFrame,
    s: dict[str, Any],
    fails: dict[str, pd.DataFrame],
    primary: str,
    methods: list[str],
    has_clip: bool,
    clip_model: str,
    results_dir: Path,
) -> None:
    """Generate results/ANALYSIS.md from the computed numbers."""
    P = primary.upper()
    L: list[str] = ["# SketchesToLife — Quantitative Analysis", ""]
    if s["mock"]:
        L += [
            "> [!WARNING]",
            "> **These numbers come from CPU *mock* outputs, not diffusion models.** The mock",
            "> generator pastes the cleaned sketch onto a colour gradient, so structure scores are",
            "> artificially perfect and identical across settings, and runtime/VRAM are ~0.",
            "> Re-run `scripts/run_experiments.py` on Colab, then `scripts/analyze.py`.",
            "> **Do not quote anything in this file in the report until this banner is gone.**",
            "",
        ]
    L += [
        f"*Auto-generated by `scripts/analyze.py` from {s['n']} generated images "
        f"({', '.join(s['sketches'])}). Edit the script, not this file.*",
        "",
        "## How to read the metrics",
        "",
        f"- **Edge F1 / recall / precision** — edges extracted from the generated image "
        f"({', '.join(m.upper() for m in methods)}) vs. the cleaned sketch, both thinned to "
        f"1-px lines at {EVAL_RESOLUTION}×{EVAL_RESOLUTION}; a pixel matches if the other map "
        f"has an edge within τ = {DEFAULT_TAU:g} px. **Recall** = share of sketch lines that "
        "survive (the core “was the sketch preserved” number). **Precision** = share of image "
        "edges that lie on a sketch line — generated texture (grass, reflections) legitimately "
        "lowers it, so precision is *expected* to be well below 1.",
        "- **Chamfer** — mean distance (px) to the nearest matching edge; *sketch→gen* "
        "grows when lines are dropped, *gen→sketch* grows when structure is invented.",
        "- **Edge SSIM** — SSIM between the two edge maps after a small blur "
        "(shift-tolerant overlap).",
        f"- **CLIPScore** — 100·cos(image, prompt) with `{clip_model}` against the content "
        "prompt (style suffixes excluded; `clip_score_full` in the CSV includes them)."
        if has_clip
        else "- **CLIPScore** — not computed in this run (`--no-clip`).",
        "- **Runtime / peak VRAM** — measured per image during generation "
        "(`torch.cuda.max_memory_allocated` after a peak reset; includes resident weights).",
        "- **Aesthetic score — deliberately not computed.** The LAION predictor needs another "
        "1.7 GB CLIP-L model plus weights with no verified Hugging Face ID, is trained on photos "
        "(unreliable for watercolour/concept-art presets), and quality is not one of our research "
        "questions; the side-by-side grids cover it qualitatively.",
        f"- Headline structure metric throughout: **{P}**.",
        "",
    ]

    L += ["## Key numbers", ""]
    if "e1" in s:
        L += [
            "### Backends (E1 — same sketches, prompt, seed 42)",
            "",
            md_table(
                s["e1"],
                {
                    "Runtime s": "{:.1f}",
                    "Peak VRAM MB": "{:.0f}",
                    "Chamfer px": "{:.1f}",
                    "CLIP": "{:.1f}",
                },
            ),
            "",
            "![backend comparison](backend_comparison.png)",
            "",
        ]
    if "e2" in s:
        e2 = s["e2"].reset_index()
        e2.columns = [
            "Scale",
            "F1",
            "Recall",
            "Precision",
            "Chamfer s→g px",
            "Chamfer g→s px",
            "Edge density",
        ] + (["CLIP"] if has_clip else [])
        L += [
            "### Conditioning scale (E2 — SD1.5 scribble)",
            "",
            md_table(
                e2,
                {
                    "Scale": "{:g}",
                    "Chamfer s→g px": "{:.1f}",
                    "Chamfer g→s px": "{:.1f}",
                    "Edge density": "{:.2f}",
                    "CLIP": "{:.1f}",
                },
            ),
            "",
            "*Edge density = generated edge pixels ÷ sketch edge pixels. An image that simply "
            "redraws the strokes already scores ≈2 (Canny finds both borders of a stroke), so "
            "compare density across scales rather than against 1.*",
            "",
            "![scale sweep](scale_sweep.png)",
            "",
        ]

    L += [
        "## Best settings per backend",
        "",
        "Rule: among the settings that keep edge F1 within 90 % of that backend's best, pick "
        "the most on-prompt one (highest CLIP)."
        if has_clip
        else "Rule: highest edge F1 (no CLIP in this run).",
        "",
        md_table(s["best"], {"CLIP": "{:.1f}"}),
        "",
    ]
    if (s["best"]["Configs tried"] == 1).any():
        L += [
            "Backends with a single configuration were only run at their defaults in E1, so "
            "their “best” is not tuned — a fair caveat when comparing them with SD1.5 scribble, "
            "which was swept.",
            "",
        ]

    if "e2" in s:
        sc, rec = s["e2"], s["e2"][f"{primary}_recall"]
        lo, hi = sc.index.min(), sc.index.max()
        L += [
            "## Where structure breaks down",
            "",
            f"- Sketch-line recall goes from **{rec.loc[lo]:.2f}** at scale {lo:g} to "
            f"**{rec.loc[hi]:.2f}** at {hi:g} (Spearman ρ scale↔F1 = "
            f"{s['corr_scale_f1']:+.2f}).",
            (
                "- Recall first reaches 90 % of its maximum at "
                f"**scale {s['scale_struct_ok']:g}**; "
                "below that, the sketch is treated as a suggestion and lines are dropped"
                if s["scale_struct_ok"] > lo
                else f"- Recall is already within 90 % of its maximum at the lowest scale tested "
                f"({lo:g}), so no breakdown point lies inside the swept range"
            )
            + f" (Chamfer sketch→gen {sc[f'{primary}_chamfer_s2g'].loc[lo]:.1f} px at {lo:g} vs "
            f"{sc[f'{primary}_chamfer_s2g'].loc[hi]:.1f} px at {hi:g}).",
            f"- Edge F1 peaks at **scale {s['scale_best_f1']:g}**.",
        ]
        if has_clip:
            clip = sc["clip_score"]
            L += [
                f"- CLIP peaks at **scale {s['scale_best_clip']:g}** ({clip.max():.1f}) and is "
                f"{clip.loc[hi]:.1f} at {hi:g} (ρ scale↔CLIP = {s['corr_scale_clip']:+.2f}). "
                + (
                    "Pushing the scale past the point where structure is already locked costs "
                    "prompt alignment without buying fidelity."
                    if s["scale_best_clip"] < hi
                    else "In this run alignment did not drop at high scale."
                )
            ]
        L += [
            "",
            "Per-sketch recall by scale (which drawings lose their lines first):",
            "",
            md_table(
                s["e2_per_sketch"]
                .reset_index()
                .rename(columns=lambda c: f"{c:g}" if isinstance(c, float) else c)
            ),
            "",
        ]
    if "corr_f1_clip" in s:
        L += [
            f"Across all {s['n']} images the fidelity/alignment correlation is "
            f"ρ = {s['corr_f1_clip']:+.2f} "
            + (
                "(negative: a real trade-off)."
                if s["corr_f1_clip"] < -0.1
                else "(weak: fidelity and alignment are not strongly in tension here)."
            ),
            "",
            "![trade-off](tradeoff_scatter.png)",
            "",
        ]

    L += ["## Failure cases (auto-selected — open the images to confirm)", ""]
    fmt = {
        "conditioning_scale": "{:g}",
        f"{primary}_chamfer_s2g": "{:.1f}",
        f"{primary}_chamfer_g2s": "{:.1f}",
        "clip_score": "{:.1f}",
    }

    def _tbl(d: pd.DataFrame) -> str:
        d = d.copy()
        d["image_path"] = [f"[img]({_rel(p, results_dir)})" for p in d["image_path"]]
        return md_table(d, fmt)

    L += [
        "### Ignored lines (lowest sketch recall)",
        "",
        _tbl(fails["ignored"]),
        "",
        "**Why:** ControlNet's output enters the frozen UNet as residuals added after zero "
        "convolutions, multiplied by `conditioning_scale`. When that residual is small relative "
        "to the text cross-attention signal, the UNet follows its prior for the prompt "
        "(“a cabin”) and only loosely respects where the lines are; thin or isolated strokes "
        "(chimney smoke, whiskers) carry the least signal through the 8×-downsampled latent "
        "and vanish first.",
        "",
        "### Invented structure (highest gen→sketch Chamfer)",
        "",
        _tbl(fails["hallucinated"]),
        "",
        "**Why:** large blank areas of the sketch give ControlNet no constraint, so the base "
        "model fills them with prompt-driven detail (forests, sky texture, reflections). "
        "This is often desirable — it is where realism comes from — but it lowers precision.",
        "",
    ]
    if "over" in fails:
        L += [
            "### Over-constrained (scale ≥ 1.2 and below the sketch's median CLIP)",
            "",
            _tbl(fails["over"]) if not fails["over"].empty else "_None in this run._",
            "",
            "**Why:** at high scale the residuals dominate at *every* denoising step, including "
            "the late steps that normally render texture. The model then reproduces the strokes "
            "themselves — dark seams, outlined panels, flat cartoon shading — instead of "
            "treating them as boundaries, and semantic content from the prompt loses out. "
            "Lowering `control_guidance_end` (E4) releases the late steps and is the usual fix.",
            "",
        ]
    L += [
        "### Text and hands",
        "",
        "Not measured: none of the three benchmark prompts contain legible text or hands, so "
        "this failure mode is not exercised by the data. The expected behaviour, for the "
        "report: a scribble map gives only *where* edges are, not *what* they are, so a "
        "five-finger outline has no constraint on joint topology and SD 1.5 (64×64 latent) "
        "cannot resolve small glyphs; ControlNet can fix a hand's silhouette but not its "
        "anatomy. Adding a hand / sign sketch would be needed to test this.",
        "",
        "## Metric limitations",
        "",
        "- Edge metrics on generated images reward line-like renderings: a flat cartoon can "
        "outscore a photo with equally correct structure, so read F1 together with CLIP and "
        "the grids.",
        "- One seed per setting except E8; differences of a few F1 points are within seed "
        "noise — check E8 before claiming them.",
        "- CLIP measures subject presence, not style quality or realism.",
    ]
    (results_dir / "ANALYSIS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    logger.info("Saved %s", results_dir / "ANALYSIS.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--outputs-dir", type=Path, default=_ROOT / "outputs" / "experiments"
    )
    ap.add_argument("--sketches-dir", type=Path, default=_ROOT / "sketches")
    ap.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    ap.add_argument(
        "--edge-methods",
        default="hed,pidinet,canny",
        help=f"Comma list from {EDGE_METHODS}; the first is the headline metric.",
    )
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument(
        "--no-clip", action="store_true", help="Skip CLIP (no model download)."
    )
    ap.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    ap.add_argument(
        "--include-mock",
        action="store_true",
        help="Keep mock outputs even when real outputs exist.",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    methods = [m.strip().lower() for m in args.edge_methods.split(",") if m.strip()]
    bad = set(methods) - set(EDGE_METHODS)
    if bad:
        ap.error(f"unknown edge method(s): {sorted(bad)}")
    primary, has_clip = methods[0], not args.no_clip
    args.results_dir.mkdir(parents=True, exist_ok=True)

    records = find_records(args.outputs_dir, args.sketches_dir)
    if not records:
        sys.exit(f"No generated images with *_meta.json found under {args.outputs_dir}")
    n_mock = sum(r["meta"].get("mock", False) for r in records)
    if 0 < n_mock < len(records) and not args.include_mock:
        logger.warning("Dropping %d mock outputs (real outputs present).", n_mock)
        records = [r for r in records if not r["meta"].get("mock", False)]
    elif n_mock == len(records):
        logger.warning(
            "ALL %d outputs are CPU mocks - numbers are a smoke test only.", n_mock
        )

    clip = ClipScorer(args.clip_model) if has_clip else None
    df = score_records(records, MetricConfig(tuple(methods), args.tau), clip)
    if not has_clip:
        df["clip_score"] = np.nan
    df.to_csv(args.results_dir / "metrics.csv", index=False, float_format="%.4f")
    logger.info("Saved %s (%d rows)", args.results_dir / "metrics.csv", len(df))

    e1, e2 = df[df["experiment"] == "E1"], df[df["experiment"] == "E2"]
    if not e2.empty:
        plot_scale_sweep(e2, methods, has_clip, args.results_dir / "scale_sweep.png")
    if not e1.empty:
        plot_backend_comparison(
            e1, primary, has_clip, args.results_dir / "backend_comparison.png"
        )
    if has_clip:
        plot_tradeoff(df, primary, args.results_dir / "tradeoff_scatter.png")

    s = summarise(df, primary, has_clip)
    write_report(
        df,
        s,
        failure_cases(df, primary, has_clip),
        primary,
        methods,
        has_clip,
        args.clip_model,
        args.results_dir,
    )

    print(
        "\n=== Key numbers"
        + ("  [MOCK DATA - NOT VALID RESULTS]" if s["mock"] else "")
        + " ==="
    )
    if "e1" in s:
        print(s["e1"].round(3).to_string(index=False))
    if "e2" in s:
        print("\nE2 scale sweep (means):")
        print(s["e2"].round(3).to_string())
    print("\nBest settings:")
    print(s["best"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
