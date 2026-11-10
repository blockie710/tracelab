"""Ingest real traces from disk.

The proof of concept only ever saw its own simulator. That is the single biggest
thing between a demo and something a team can adopt: nobody can point it at
their own data.

Two input formats, both boring on purpose:

JSONL — one trace per line::

    {"id": "match-8812-a", "label": "human", "tick_hz": 60,
     "yaw": [0.0, 0.4, ...], "pitch": [...],
     "target_yaw": [...], "target_pitch": [...],
     "visible": [true, true, false, ...]}

CSV — long format, one row per tick, grouped by ``trace_id``::

    trace_id,tick,yaw,pitch,target_yaw,target_pitch,visible,label
    a,0,0.0,0.0,12.4,1.1,true,human
    a,1,0.2,0.0,12.4,1.1,true,human

Validation is strict and loud. A silently-accepted malformed trace is worse than
a crash, because it becomes a wrong number in a report someone trusts.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Iterable, List, Sequence

from .simulate import Trace

KNOWN_LABELS = ("human", "snap", "smoothed", "unknown")
MIN_TICKS = 30


class IngestError(ValueError):
    """Raised with the offending trace id and field, never just 'invalid data'."""


def _as_floats(values: Sequence, field: str, tid: str) -> List[float]:
    out = []
    for i, v in enumerate(values):
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            raise IngestError(f"trace {tid!r}: {field}[{i}] is not numeric ({v!r})")
    return out


def _as_bools(values: Sequence, field: str, tid: str) -> List[bool]:
    out = []
    for i, v in enumerate(values):
        if isinstance(v, bool):
            out.append(v)
        elif isinstance(v, (int, float)):
            out.append(bool(v))
        elif isinstance(v, str):
            out.append(v.strip().lower() in ("1", "true", "yes", "y"))
        else:
            raise IngestError(f"trace {tid!r}: {field}[{i}] is not boolean ({v!r})")
    return out


def collect_warnings(trace: Trace, tick_hz: int = 60) -> List[str]:
    """Plausible-but-fast motion. Reported, never fatal.

    A snapped control signal legitimately covers tens of degrees in one tick.
    An ingest layer that rejects that is rejecting the thing the tool exists to
    measure, so fast motion is a warning; only physically impossible motion is
    an error.
    """
    tid = getattr(trace, "trace_id", "?")
    out = []
    fast_limit = 1200.0 / tick_hz
    fast = [i for i in range(1, trace.n) if abs(trace.yaw[i] - trace.yaw[i - 1]) > fast_limit]
    if fast:
        out.append(f"trace {tid!r}: {len(fast)} tick(s) faster than {fast_limit:.0f} deg/tick")
    if trace.n and max(abs(v) for v in trace.yaw) > 720:
        out.append(f"trace {tid!r}: yaw exceeds +/-720 deg — check units")
    return out


def validate(trace: Trace, tick_hz: int = 60) -> Trace:
    """Structural checks only. Returns the trace for chaining.

    Errors here mean the file is wrong, not that the behaviour is unusual.
    """
    tid = getattr(trace, "trace_id", "?")
    n = trace.n
    if n < MIN_TICKS:
        raise IngestError(f"trace {tid!r}: only {n} ticks, need at least {MIN_TICKS}")
    if not (len(trace.pitch) == len(trace.target_yaw) == len(trace.target_pitch) == n):
        raise IngestError(
            f"trace {tid!r}: length mismatch — yaw={n}, pitch={len(trace.pitch)}, "
            f"target_yaw={len(trace.target_yaw)}, target_pitch={len(trace.target_pitch)}"
        )
    if trace.visible and len(trace.visible) != n:
        raise IngestError(f"trace {tid!r}: visible has {len(trace.visible)} flags for {n} ticks")

    for field_name, series in (
        ("yaw", trace.yaw), ("pitch", trace.pitch),
        ("target_yaw", trace.target_yaw), ("target_pitch", trace.target_pitch),
    ):
        for i, v in enumerate(series):
            if not math.isfinite(v):
                raise IngestError(f"trace {tid!r}: {field_name}[{i}] is not finite ({v!r})")

    # Physically impossible: any real control solution moves under ~60 deg/ms.
    impossible = 2_000_000.0 / tick_hz
    for i in range(1, n):
        step = abs(trace.yaw[i] - trace.yaw[i - 1])
        if step > impossible:
            raise IngestError(
                f"trace {tid!r}: yaw jumps {step:.0f} deg in one tick at index {i} — "
                "likely radians, degrees*100, or a desync"
            )
    return trace


def from_records(records: Iterable[Dict]) -> List[Trace]:
    traces: List[Trace] = []
    for rec in records:
        tid = str(rec.get("id") or rec.get("trace_id") or f"row{len(traces)}")
        label = str(rec.get("label") or rec.get("kind") or "unknown").strip().lower()
        if label not in KNOWN_LABELS:
            raise IngestError(
                f"trace {tid!r}: unknown label {label!r}; expected one of {KNOWN_LABELS}"
            )
        tick_hz = int(rec.get("tick_hz") or 60)
        yaw = _as_floats(rec["yaw"], "yaw", tid)
        pitch = _as_floats(rec["pitch"], "pitch", tid)
        tyaw = _as_floats(rec["target_yaw"], "target_yaw", tid)
        tpitch = _as_floats(rec["target_pitch"], "target_pitch", tid)
        vis = rec.get("visible")
        visible = _as_bools(vis, "visible", tid) if vis is not None else [True] * len(yaw)

        t = Trace(kind=label, yaw=yaw, pitch=pitch, target_yaw=tyaw,
                  target_pitch=tpitch, visible=visible)
        setattr(t, "trace_id", tid)
        setattr(t, "tick_hz", tick_hz)
        traces.append(validate(t, tick_hz))
    return traces


def load_jsonl(path: str) -> List[Trace]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise IngestError(f"{os.path.basename(path)}:{lineno}: invalid JSON ({e.msg})")
    return from_records(records)


def load_csv(path: str) -> List[Trace]:
    grouped: Dict[str, List[Dict]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"trace_id", "yaw", "pitch", "target_yaw", "target_pitch"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise IngestError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            grouped.setdefault(row["trace_id"], []).append(row)

    records = []
    for tid, rows in grouped.items():
        rows.sort(key=lambda r: int(r.get("tick") or 0))
        records.append(
            {
                "id": tid,
                "label": rows[0].get("label") or "unknown",
                "tick_hz": rows[0].get("tick_hz") or 60,
                "yaw": [r["yaw"] for r in rows],
                "pitch": [r["pitch"] for r in rows],
                "target_yaw": [r["target_yaw"] for r in rows],
                "target_pitch": [r["target_pitch"] for r in rows],
                "visible": [r.get("visible", "true") for r in rows],
            }
        )
    return from_records(records)


def load(path: str) -> List[Trace]:
    """Dispatch on extension. Supports .jsonl / .ndjson / .json / .csv."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".ndjson"):
        return load_jsonl(path)
    if ext == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return from_records(data if isinstance(data, list) else [data])
    if ext == ".csv":
        return load_csv(path)
    raise IngestError(f"unsupported extension {ext!r}; use .jsonl, .json or .csv")


def to_jsonl(traces: Sequence[Trace], path: str) -> None:
    """Write traces back out — used by `tracelab export` and by tests."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for t in traces:
            rec = {
                "id": getattr(t, "trace_id", f"{t.kind}-{id(t)}"),
                "label": t.kind,
                "tick_hz": getattr(t, "tick_hz", 60),
                **t.as_dict(),
            }
            rec.pop("kind", None)
            fh.write(json.dumps(rec) + "\n")
