"""Same seed -> same pixels, even across processes with different hash salts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SNIPPET = """
import hashlib, numpy as np
from sketch2image.pipelines import SketchToImageGenerator
g = SketchToImageGenerator(mock=True)
imgs, _ = g.generate("sketches/sketch_01.png", "a cabin", seed=42, num_images_per_prompt=2)
print("|".join(hashlib.sha256(np.asarray(i).tobytes()).hexdigest() for i in imgs))
"""


def _digest(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_mock_generation_is_identical_across_processes() -> None:
    # Different PYTHONHASHSEEDs: a str-hash-based colour would differ here.
    assert _digest("1") == _digest("2")


def test_mock_images_in_a_batch_differ() -> None:
    a, b = _digest("0").split("|")
    assert a != b
