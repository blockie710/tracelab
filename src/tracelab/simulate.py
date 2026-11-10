"""Synthetic aim-trace generator.

This module is the ground truth for the demo. It produces the same class of
signal a game server already collects for free: view angles at 60 Hz, plus a
per-tick visibility flag. Three controllers are simulated:

    human             second-order limb response, ~215 ms perceptual latency,
                      physiological tremor, overshoot and corrective reversals,
                      and it loses the target while the target is occluded
    snap            near-critically-damped snap with one-tick latency, near
                      zero residual error, and it keeps tracking through cover
    smoothed  slower snap plus injected tremor and micro-jitter, the
                      smoothing real positives use to defeat naive detectors

Nothing here is captured from a live system. It is a physically motivated
simulation, which is the point: a classifier can be measured honestly before it
is ever pointed at production traffic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List

TICK_HZ = 60
DT = 1.0 / TICK_HZ

KINDS = ("human", "snap", "smoothed")


@dataclass
class Trace:
    """One engagement window for one player."""

    kind: str
    yaw: List[float] = field(default_factory=list)
    pitch: List[float] = field(default_factory=list)
    target_yaw: List[float] = field(default_factory=list)
    target_pitch: List[float] = field(default_factory=list)
    visible: List[bool] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.yaw)

    def error(self) -> List[float]:
        """Angular error to target, in degrees, per tick."""
        return [
            math.hypot(ty - ay, tp - ap)
            for ay, ap, ty, tp in zip(self.yaw, self.pitch, self.target_yaw, self.target_pitch)
        ]

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "yaw": [round(v, 4) for v in self.yaw],
            "pitch": [round(v, 4) for v in self.pitch],
            "target_yaw": [round(v, 4) for v in self.target_yaw],
            "target_pitch": [round(v, 4) for v in self.target_pitch],
            "visible": self.visible,
        }


# --- target motion -------------------------------------------------------


def _target_path(rng: random.Random, n: int, mode: str):
    """Return (yaw, pitch) lists of the target's world-space angle."""
    if mode == "flick":
        # Discrete repositions: a target popping between cover.
        yaw = [0.0] * n
        pitch = [0.0] * n
        y = rng.uniform(-35, 35)
        p = rng.uniform(-12, 12)
        next_jump = rng.randint(25, 55)
        for i in range(n):
            if i >= next_jump:
                y = rng.uniform(-38, 38)
                p = rng.uniform(-14, 14)
                next_jump = i + rng.randint(30, 70)
            yaw[i] = y
            pitch[i] = p
        return yaw, pitch

    # Smooth strafing target: two sinusoids plus slow drift.
    a_y = rng.uniform(6, 22)
    a_p = rng.uniform(1.5, 7)
    w_y = rng.uniform(1.6, 4.2)
    w_p = rng.uniform(1.1, 2.8)
    ph_y = rng.uniform(0, math.tau)
    ph_p = rng.uniform(0, math.tau)
    drift = rng.uniform(-6, 6)
    yaw = [a_y * math.sin(w_y * i * DT + ph_y) + drift * i * DT for i in range(n)]
    pitch = [a_p * math.sin(w_p * i * DT + ph_p) for i in range(n)]
    return yaw, pitch


def _visibility(rng: random.Random, n: int, p_hidden: float):
    """Contiguous occlusion windows, the way cover actually works."""
    vis = [True] * n
    i = rng.randint(10, 40)
    while i < n:
        length = rng.randint(12, 45)
        for j in range(i, min(n, i + length)):
            vis[j] = False
        i += length + rng.randint(30, 90)
    if p_hidden > 0:
        for j in range(n):
            if not vis[j]:
                continue
            if rng.random() < p_hidden * 0.02:
                vis[j] = False
    return vis


# --- controller ----------------------------------------------------------


def _respond(
    target: List[float],
    visible: List[bool],
    rng: random.Random,
    *,
    wn: float,
    zeta: float,
    latency: int,
    noise: float,
    tremor_amp: float,
    tremor_hz: float,
    sees_through_cover: bool,
    tremor_phase: float,
) -> List[float]:
    """Second-order spring-damper limb controller driven by perceived target.

    wn is the natural frequency in rad/s and zeta the damping ratio, which is
    what separates a human limb (under-damped, slow) from a snap (critically
    damped, fast).
    """
    n = len(target)
    aim = [0.0] * n
    vel = 0.0
    last_seen = target[0]

    for i in range(n):
        j = max(0, i - latency)
        if visible[j] or sees_through_cover:
            perceived = target[j]
            if visible[j]:
                last_seen = target[j]
        else:
            # Human keeps aiming where the target was last seen; the target
            # moves on, so error grows.
            perceived = last_seen

        err = perceived - aim[i - 1] if i else perceived
        acc = wn * wn * err - 2.0 * zeta * wn * vel
        vel += acc * DT
        step = vel * DT
        trem = tremor_amp * math.sin(math.tau * tremor_hz * i * DT + tremor_phase)
        aim[i] = aim[i - 1] + step + trem + (rng.gauss(0.0, noise) if noise else 0.0)
    return aim


PARAMS = {
    #                    wn    zeta  lat  noise  trem_amp trem_hz cover
    "human":            (7.0, 0.52, 13, 0.06, 0.30, 9.5, False),
    "snap":           (30.0, 1.02, 1, 0.004, 0.0, 11.0, True),
    "smoothed": (12.0, 0.94, 3, 0.06, 0.18, 10.5, True),
}


def simulate_trace(kind: str, rng: random.Random | None = None, ticks: int = 180, mode: str = "track") -> Trace:
    """Generate one labelled engagement window."""
    if kind not in PARAMS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    rng = rng or random.Random()
    tyaw, tpitch = _target_path(rng, ticks, mode)
    vis = _visibility(rng, ticks, p_hidden=0.0)

    wn, zeta, lat, noise, tamp, thz, cover = PARAMS[kind]
    # Per-player jitter in the physiological parameters.
    wn *= rng.uniform(0.85, 1.15)
    zeta *= rng.uniform(0.9, 1.1)
    lat = max(0, lat + rng.randint(-1, 1))
    phase = rng.uniform(0, math.tau)

    yaw = _respond(tyaw, vis, rng, wn=wn, zeta=zeta, latency=lat, noise=noise,
                   tremor_amp=tamp, tremor_hz=thz, sees_through_cover=cover,
                   tremor_phase=phase)
    pitch = _respond(tpitch, vis, rng, wn=wn * rng.uniform(0.9, 1.1), zeta=zeta,
                     latency=lat, noise=noise * 0.8, tremor_amp=tamp * 0.6,
                     tremor_hz=thz * 1.05, sees_through_cover=cover,
                     tremor_phase=phase + 1.1)
    return Trace(kind=kind, yaw=yaw, pitch=pitch, target_yaw=tyaw,
                 target_pitch=tpitch, visible=vis)


def simulate_dataset(n_per_kind: int, seed: int = 0, ticks: int = 180, modes=("track", "flick")):
    """Balanced dataset across kinds and engagement modes."""
    rng = random.Random(seed)
    out = []
    for kind in KINDS:
        for _ in range(n_per_kind):
            mode = rng.choice(list(modes))
            out.append(simulate_trace(kind, rng, ticks=ticks, mode=mode))
    rng.shuffle(out)
    return out
