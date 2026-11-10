"""Vision front-end.

The detector's inputs are control angles. A service has those. A recording, a
capture, or a screen share does not -- it has pixels. This module closes that
gap so the same detector can run on footage:

    frames -> locate crosshair -> locate target -> convert screen offset to
    angular error -> identical feature pipeline

The frame parser here is deliberately simple (colour thresholds on a synthetic
render) because the point of a proof of concept is to prove the *plumbing*.
Production swaps `detect()` for a learned detector plus object tracking and
optical-flow camera estimation; everything downstream is unchanged.

Run `tracelab vision` to see the round-trip error for yourself.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .simulate import Trace

HFOV_DEG = 80.0
SIZE = 320


def render_frame(trace: Trace, i: int, size: int = SIZE):
    """Render one synthetic first-person frame with a target and a crosshair.

    Green cross = crosshair, red rectangle = target. White-on-grey background
    so threshold detection has to cope with texture rather than a flat fill.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (28, 30, 34))
    d = ImageDraw.Draw(img)

    # Some scenery so the background is not a single colour.
    for k in range(0, size, 16):
        shade = 34 + (k // 16) % 5 * 3
        d.rectangle([0, k, size, k + 15], fill=(shade, shade + 2, shade + 6))
    d.rectangle([size * 0.1, size * 0.55, size * 0.32, size * 0.8], fill=(52, 50, 46))
    d.rectangle([size * 0.68, size * 0.45, size * 0.9, size * 0.75], fill=(46, 44, 42))

    # Target projected into screen space from the angular offset.
    def project(ang_yaw: float, ang_pitch: float) -> Tuple[float, float]:
        x = size / 2 + (ang_yaw / HFOV_DEG) * size
        y = size / 2 - (ang_pitch / HFOV_DEG) * size
        return x, y

    tx, ty = project(trace.target_yaw[i] - trace.yaw[i], trace.target_pitch[i] - trace.pitch[i])
    if -20 <= tx <= size + 20 and -20 <= ty <= size + 20:
        d.rectangle([tx - 9, ty - 22, tx + 9, ty + 22], fill=(196, 64, 52))

    # Crosshair: saturated green so it survives compression better than a
    # thin white line would.
    cx = cy = size / 2
    d.rectangle([cx - 8, cy - 1, cx + 8, cy + 1], fill=(0, 235, 90))
    d.rectangle([cx - 1, cy - 8, cx + 1, cy + 8], fill=(0, 235, 90))
    return img


def _centroid(pixels, size: int, key) -> Tuple[float, float] | None:
    xs = ys = n = 0
    for y in range(size):
        for x in range(size):
            r, g, b = pixels[x, y]
            if key(r, g, b):
                xs += x
                ys += y
                n += 1
    if not n:
        return None
    return xs / n, ys / n


def detect(img) -> Tuple[Tuple[float, float] | None, Tuple[float, float] | None]:
    """Return (crosshair_xy, target_xy) from one frame."""
    size = img.size[0]
    px = img.load()
    cross = _centroid(px, size, lambda r, g, b: g > 170 and r < 90 and b < 140)
    target = _centroid(px, size, lambda r, g, b: r > 140 and g < 110 and b < 100)
    return cross, target


def frame_to_error(cross, target, size: int = SIZE) -> float:
    """Angular error in degrees between crosshair and target centroids."""
    if cross is None or target is None:
        return 180.0
    dx = (target[0] - cross[0]) / size * HFOV_DEG
    dy = (target[1] - cross[1]) / size * HFOV_DEG
    return math.hypot(dx, dy)


def recover_errors(trace: Trace, stride: int = 3, size: int = SIZE) -> List[float]:
    """Run the whole pixel->degrees path over a trace's rendered frames."""
    out = []
    for i in range(0, trace.n, stride):
        img = render_frame(trace, i, size)
        cross, target = detect(img)
        out.append(frame_to_error(cross, target, size))
    return out


def round_trip_check(trace: Trace, stride: int = 3) -> dict:
    """Compare vision-recovered error against the ground-truth signal."""
    truth = trace.error()[::stride]
    got = recover_errors(trace, stride)
    pairs = [(t, g) for t, g in zip(truth, got) if g < 90.0]
    if not pairs:
        return {"n": 0, "mae_deg": float("nan"), "corr": float("nan")}
    mae = sum(abs(t - g) for t, g in pairs) / len(pairs)
    mt = sum(t for t, _ in pairs) / len(pairs)
    mg = sum(g for _, g in pairs) / len(pairs)
    cov = sum((t - mt) * (g - mg) for t, g in pairs)
    vt = math.sqrt(sum((t - mt) ** 2 for t, _ in pairs))
    vg = math.sqrt(sum((g - mg) ** 2 for _, g in pairs))
    corr = cov / (vt * vg) if vt and vg else 0.0
    return {"n": len(pairs), "mae_deg": mae, "corr": corr}


def save_sample_frames(trace: Trace, indices: Sequence[int], out_dir: str) -> List[str]:
    """Write PNGs so a human can eyeball what the parser was given."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in indices:
        p = os.path.join(out_dir, f"frame_{i:03d}.png")
        render_frame(trace, i).save(p)
        paths.append(p)
    return paths
