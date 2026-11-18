"""Create 3 hand-drawn style sample sketches for SketchesToLife testing.

Generates:
1. sketch_01.png: Mountain cabin with pine trees and rising smoke.
2. sketch_02.png: Sports coupe car with wheels and headlights.
3. sketch_03.png: Whimsical cat with spectacles.

Includes realistic pencil stroke texture, paper grain, and illumination gradients
to test the preprocessing cleaning and normalisation pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def add_paper_noise_and_gradient(canvas: np.ndarray) -> np.ndarray:
    """Add subtle paper grain, vignetting, and lighting gradient to simulate phone photo."""
    h, w = canvas.shape[:2]
    # 1. Subtle paper grain noise
    noise = np.random.normal(0, 7, (h, w)).astype(np.float32)

    # 2. Uneven lighting gradient (brighter at top-left, shadowed at bottom-right)
    y, x = np.mgrid[0:h, 0:w]
    gradient = ((x / w) * 0.15 + (y / h) * 0.25) * 50

    # 3. Vignette shadow in corners
    center_x, center_y = w / 2, h / 2
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = (dist_from_center / max_dist) * 35

    degraded = canvas.astype(np.float32) - gradient - vignette + noise
    return np.clip(degraded, 0, 255).astype(np.uint8)


def create_sketch_01_cabin() -> Image.Image:
    """Sketch 01: Mountain cabin with pine trees, mountains, and sun."""
    w, h = 600, 500
    canvas = np.full((h, w), 245, dtype=np.uint8)

    # Draw mountains in background
    cv2.polylines(canvas, [np.array([[20, 320], [180, 120], [330, 320]], np.int32)], False, 40, 3, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array([[250, 320], [420, 80], [580, 320]], np.int32)], False, 45, 3, cv2.LINE_AA)
    # Sun behind mountain
    cv2.circle(canvas, (260, 110), 35, 60, 2, cv2.LINE_AA)

    # Cabin body
    cv2.rectangle(canvas, (200, 260), (380, 420), 30, 4, cv2.LINE_AA)
    # Roof (triangle)
    cv2.polylines(canvas, [np.array([[170, 260], [290, 170], [410, 260]], np.int32)], True, 25, 4, cv2.LINE_AA)
    # Chimney
    cv2.rectangle(canvas, (330, 170), (360, 240), 35, 3, cv2.LINE_AA)
    # Smoke curls
    for i, offset_y in enumerate([150, 120, 90, 60]):
        cv2.ellipse(canvas, (345 + (i * 12), offset_y), (15 + i * 3, 10), 0, 0, 220, 55, 2, cv2.LINE_AA)
    # Door and window
    cv2.rectangle(canvas, (270, 330), (320, 420), 40, 3, cv2.LINE_AA)
    cv2.circle(canvas, (312, 380), 3, 30, -1)
    cv2.rectangle(canvas, (220, 300), (255, 340), 45, 2, cv2.LINE_AA)
    cv2.line(canvas, (237, 300), (237, 340), 45, 2)
    cv2.line(canvas, (220, 320), (255, 320), 45, 2)

    # Pine trees
    def draw_pine(base_x: int, base_y: int, scale: float = 1.0):
        for tier in range(4):
            y_top = int(base_y - (tier + 1) * 35 * scale)
            y_bot = int(base_y - tier * 25 * scale)
            span = int((4 - tier) * 18 * scale)
            pts = np.array([[base_x, y_top], [base_x - span, y_bot], [base_x + span, y_bot]], np.int32)
            cv2.polylines(canvas, [pts], True, 35, 3, cv2.LINE_AA)
        cv2.line(canvas, (base_x, base_y), (base_x, int(base_y - 20 * scale)), 40, 4)

    draw_pine(100, 440, 1.1)
    draw_pine(150, 460, 0.9)
    draw_pine(470, 430, 1.2)
    draw_pine(530, 450, 0.8)

    # Ground line
    cv2.line(canvas, (10, 440), (590, 440), 35, 3, cv2.LINE_AA)

    # Add realistic scanning/photo flaws
    realistic = add_paper_noise_and_gradient(canvas)
    return Image.fromarray(realistic)


def create_sketch_02_car() -> Image.Image:
    """Sketch 02: Sleek sports car."""
    w, h = 600, 450
    canvas = np.full((h, w), 242, dtype=np.uint8)

    # Car body contour
    pts = np.array([
        [60, 290],   # Front bumper
        [100, 270],  # Hood start
        [200, 240],  # Hood end / windshield base
        [270, 160],  # Windshield top
        [410, 160],  # Roof end
        [490, 230],  # Rear window
        [550, 240],  # Trunk / spoiler
        [560, 290],  # Rear bumper
        [500, 290],  # Rear wheel right
        [430, 290],  # Rear wheel left
        [240, 290],  # Front wheel right
        [160, 290],  # Front wheel left
        [60, 290],   # Back to front
    ], np.int32)
    cv2.polylines(canvas, [pts], False, 30, 4, cv2.LINE_AA)

    # Front wheel
    cv2.circle(canvas, (200, 295), 45, 30, 4, cv2.LINE_AA)
    cv2.circle(canvas, (200, 295), 25, 40, 2, cv2.LINE_AA)
    cv2.circle(canvas, (200, 295), 8, 30, -1)

    # Rear wheel
    cv2.circle(canvas, (465, 295), 45, 30, 4, cv2.LINE_AA)
    cv2.circle(canvas, (465, 295), 25, 40, 2, cv2.LINE_AA)
    cv2.circle(canvas, (465, 295), 8, 30, -1)

    # Door and window line
    cv2.line(canvas, (275, 175), (210, 245), 40, 3, cv2.LINE_AA)
    cv2.line(canvas, (275, 175), (400, 175), 40, 3, cv2.LINE_AA)
    cv2.line(canvas, (400, 175), (470, 235), 40, 3, cv2.LINE_AA)
    cv2.line(canvas, (210, 245), (470, 245), 40, 3, cv2.LINE_AA)
    # Side mirror
    cv2.rectangle(canvas, (215, 235), (235, 248), 35, 2, cv2.LINE_AA)

    # Headlight & spoiler
    cv2.line(canvas, (65, 275), (100, 268), 35, 3, cv2.LINE_AA)
    cv2.line(canvas, (540, 230), (565, 220), 25, 4, cv2.LINE_AA)  # Spoiler wing

    # Ground line & shadow
    cv2.line(canvas, (30, 340), (580, 340), 50, 3, cv2.LINE_AA)
    cv2.ellipse(canvas, (330, 342), (240, 8), 0, 0, 360, 90, 3, cv2.LINE_AA)

    realistic = add_paper_noise_and_gradient(canvas)
    return Image.fromarray(realistic)


def create_sketch_03_cat() -> Image.Image:
    """Sketch 03: Whimsical cat wearing glasses."""
    w, h = 500, 550
    canvas = np.full((h, w), 248, dtype=np.uint8)

    # Cat head oval
    cv2.ellipse(canvas, (250, 280), (140, 130), 0, 0, 360, 30, 4, cv2.LINE_AA)

    # Ears
    cv2.polylines(canvas, [np.array([[150, 200], [120, 80], [210, 165]], np.int32)], False, 30, 4, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array([[350, 200], [380, 80], [290, 165]], np.int32)], False, 30, 4, cv2.LINE_AA)
    # Inner ear details
    cv2.line(canvas, (140, 170), (135, 110), 60, 2, cv2.LINE_AA)
    cv2.line(canvas, (360, 170), (365, 110), 60, 2, cv2.LINE_AA)

    # Round Spectacles (glasses)
    cv2.circle(canvas, (190, 270), 38, 25, 4, cv2.LINE_AA)
    cv2.circle(canvas, (310, 270), 38, 25, 4, cv2.LINE_AA)
    # Glasses bridge
    cv2.line(canvas, (228, 270), (272, 270), 25, 4, cv2.LINE_AA)
    # Cat eyes inside glasses
    cv2.circle(canvas, (190, 270), 12, 40, -1)
    cv2.circle(canvas, (310, 270), 12, 40, -1)
    cv2.circle(canvas, (194, 267), 4, 245, -1) # Pupil highlight
    cv2.circle(canvas, (314, 267), 4, 245, -1)

    # Nose & mouth
    pts_nose = np.array([[240, 325], [260, 325], [250, 335]], np.int32)
    cv2.fillPoly(canvas, [pts_nose], 35)
    cv2.line(canvas, (250, 335), (250, 345), 35, 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (235, 348), (15, 10), 0, 0, 180, 35, 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (265, 348), (15, 10), 0, 0, 180, 35, 2, cv2.LINE_AA)

    # Whiskers
    for offset in [-10, 0, 10]:
        cv2.line(canvas, (160, 335 + offset), (70, 325 + offset * 2), 40, 2, cv2.LINE_AA)
        cv2.line(canvas, (340, 335 + offset), (430, 325 + offset * 2), 40, 2, cv2.LINE_AA)

    # Bowtie
    cv2.polylines(canvas, [np.array([[250, 420], [210, 395], [210, 445]], np.int32)], True, 30, 3, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array([[250, 420], [290, 395], [290, 445]], np.int32)], True, 30, 3, cv2.LINE_AA)
    cv2.circle(canvas, (250, 420), 8, 30, -1)

    # Shoulders
    cv2.ellipse(canvas, (250, 490), (170, 70), 0, 180, 360, 35, 3, cv2.LINE_AA)

    realistic = add_paper_noise_and_gradient(canvas)
    return Image.fromarray(realistic)


def main() -> None:
    out_dir = Path("sketches")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample sketches in sketches/ ...")
    s1 = create_sketch_01_cabin()
    s1.save(out_dir / "sketch_01.png")
    print(f"  ✓ Saved {out_dir / 'sketch_01.png'} ({s1.size})")

    s2 = create_sketch_02_car()
    s2.save(out_dir / "sketch_02.png")
    print(f"  ✓ Saved {out_dir / 'sketch_02.png'} ({s2.size})")

    s3 = create_sketch_03_cat()
    s3.save(out_dir / "sketch_03.png")
    print(f"  ✓ Saved {out_dir / 'sketch_03.png'} ({s3.size})")


if __name__ == "__main__":
    main()
