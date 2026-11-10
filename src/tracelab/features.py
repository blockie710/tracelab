"""Feature extraction from view-angle telemetry.

Every feature here is computable from data a service already has: a two-axis
control signal per tick plus a visibility flag. Nothing instrumented on the
client, no privileged access.

The features encode *mechanism* rather than skill. An expert operator and a
scripted actor can share a reaction time; they do not easily share corrective
reversal structure, tremor periodicity, or behaviour under occlusion.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from .simulate import DT, Trace

FEATURE_NAMES = [
    "acquire_time_s",
    "peak_speed_dps",
    "peak_accel_dps2",
    "err_median_vis",
    "err_std_vis",
    "reversal_rate",
    "occl_drift",
    "cover_track_frac",
    "tremor_ratio",
    "speed_kurtosis",
    "stationary_frac",
    "overshoot_ratio",
    "err_autocorr1",
]

_EPS = 1e-9


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _std(xs: Sequence[float]) -> float:
    return math.sqrt(_var(xs))


def _tone_power(xs: Sequence[float], freq: float) -> float:
    """Goertzel power at one frequency. Cheap, and no FFT dependency."""
    n = len(xs)
    if n < 8 or not xs:
        return 0.0
    m = _mean(xs)
    w = math.tau * freq / (1.0 / DT)  # freq / sample_rate, in radians
    coeff = 2.0 * math.cos(w)
    s0 = s1 = s2 = 0.0
    for x in xs:
        s0 = (x - m) + coeff * s1 - s2
        s2, s1 = s1, s0
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return max(0.0, power) / n


def _kurtosis(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 4:
        return 0.0
    m = _mean(xs)
    v = _var(xs)
    if v < _EPS:
        return 0.0
    return sum(((x - m) / math.sqrt(v)) ** 4 for x in xs) / n


def _autocorr1(xs: Sequence[float]) -> float:
    if len(xs) < 3:
        return 0.0
    m = _mean(xs)
    d = [x - m for x in xs]
    num = sum(d[i] * d[i - 1] for i in range(1, len(d)))
    den = sum(x * x for x in d)
    return num / den if den > _EPS else 0.0


def extract(trace: Trace) -> Dict[str, float]:
    """12 mechanism features for one engagement window."""
    n = trace.n
    if n < 8:
        return {k: 0.0 for k in FEATURE_NAMES}

    err = trace.error()
    dyaw = [(trace.yaw[i] - trace.yaw[i - 1]) / DT for i in range(1, n)]
    ddyaw = [(dyaw[i] - dyaw[i - 1]) / DT for i in range(1, len(dyaw))]
    speed = [abs(v) for v in dyaw]

    vis_err = [e for e, v in zip(err, trace.visible) if v]
    occ_err = [e for e, v in zip(err, trace.visible) if not v]

    # Acquisition: first time the crosshair is within 1 degree of the target.
    acquire = n * DT
    for i, e in enumerate(err):
        if e <= 1.0:
            acquire = i * DT
            break

    # Corrective reversals: sign changes in aim velocity while roughly on
    # target. Organic control oscillates around the target; a snap does not.
    on_target = [i for i, e in enumerate(err) if e < 10.0 and i < len(dyaw)]
    reversals = 0
    for a, b in zip(on_target, on_target[1:]):
        if b == a + 1 and dyaw[a] * dyaw[b] < 0 and min(speed[a], speed[b]) > 0.5:
            reversals += 1
    rev_window = max(len(on_target), 1) * DT
    reversal_rate = reversals / rev_window

    # Behaviour under occlusion. An operator loses the target and the control
    # signal drifts off it; a scripted actor that reads state keeps it glued. Measured per occlusion window as
    # the change in error from the last visible tick to the last hidden tick.
    windows = []
    i = 0
    while i < n:
        if not trace.visible[i]:
            j = i
            while j + 1 < n and not trace.visible[j + 1]:
                j += 1
            windows.append((i, j))
            i = j + 1
        else:
            i += 1

    drifts = []
    for a, b in windows:
        before = err[a - 1] if a else err[a]
        drifts.append(err[b] - before)
    occl_drift = _median(drifts) if drifts else 0.0
    vis_med = _median(vis_err)

    hidden_idx = [i for i, v in enumerate(trace.visible) if not v]
    cover_track = (
        sum(1 for i in hidden_idx if err[i] < 2.0) / len(hidden_idx) if hidden_idx else 0.0
    )

    # Physiological tremor: periodic energy in the 6-15 Hz band, measured on
    # aim *velocity* while on target. Position error is dominated by the
    # acquisition transient, so normalising against it hides the tremor; the
    # velocity signal is where a 9-11 Hz limb oscillation actually shows up.
    seg = [dyaw[i] for i, e in enumerate(err[: len(dyaw)]) if e < 5.0]
    if len(seg) >= 16:
        tone = sum(_tone_power(seg, f) for f in (6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0))
        seg_total = _var(seg) * len(seg) / 2.0 + _EPS
        tremor_ratio = min(1.0, tone / seg_total)
    else:
        tremor_ratio = 0.0

    stationary = sum(1 for s in speed if s < 1.0) / max(len(speed), 1)

    # Overshoot: how far past the target the aim travels after first acquiring.
    first = next((i for i, e in enumerate(err) if e <= 2.0), None)
    overshoot = 0.0
    if first is not None and first + 20 < n:
        overshoot = max(err[first:first + 20]) / max(err[0], 1.0)

    return {
        "acquire_time_s": acquire,
        "peak_speed_dps": max(speed) if speed else 0.0,
        "peak_accel_dps2": max((abs(a) for a in ddyaw), default=0.0),
        "err_median_vis": vis_med,
        "err_std_vis": _std(vis_err),
        "reversal_rate": reversal_rate,
        "occl_drift": occl_drift,
        "cover_track_frac": cover_track,
        "tremor_ratio": tremor_ratio,
        "speed_kurtosis": _kurtosis(speed),
        "stationary_frac": stationary,
        "overshoot_ratio": overshoot,
        "err_autocorr1": _autocorr1(err),
    }


def extract_many(traces: Sequence[Trace]) -> List[List[float]]:
    """Matrix of features in FEATURE_NAMES order."""
    return [[extract(t)[k] for k in FEATURE_NAMES] for t in traces]
