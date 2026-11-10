# Trace input format

`tracelab` reads two formats and nothing else. Both are boring on purpose: an
integrity tool that demands a bespoke binary format is a tool nobody adopts.

## What a trace is

One **engagement window** — a contiguous slice of a two-axis control signal,
typically 2–6 seconds at 60 Hz, aligned to a single target interaction. Do not
mix windows, do not concatenate across targets, do not resample by hand.

Required series, all the same length, all in **degrees**:

| field | meaning |
|---|---|
| `yaw` | horizontal control angle per tick |
| `pitch` | vertical control angle per tick |
| `target_yaw` | horizontal angle of the tracked object |
| `target_pitch` | vertical angle of the tracked object |
| `visible` | optional bool per tick; `true` = target on screen. Defaults to all `true` |

`label` is optional for scoring, required for training, and must be one of
`human`, `snap`, `smoothed`, or `unknown`.

Minimum length: **30 ticks**. Below that the derived features are noise.

## JSONL (preferred)

```json
{"id": "match-8812-a", "label": "human", "tick_hz": 60,
 "yaw": [0.0, 0.4, 1.1], "pitch": [0.0, 0.0, 0.1],
 "target_yaw": [12.4, 12.4, 12.3], "target_pitch": [1.1, 1.1, 1.1],
 "visible": [true, true, false]}
```

One object per line. Comments (`#`) and blank lines are ignored.

## CSV (long format)

```
trace_id,tick,yaw,pitch,target_yaw,target_pitch,visible,label
a,0,0.0,0.0,12.4,1.1,true,human
a,1,0.2,0.0,12.4,1.1,true,human
```

Rows are grouped by `trace_id` and sorted by `tick`.

## Validation rules

Errors (the file is wrong):

- series of unequal length
- fewer than 30 ticks
- `NaN` or `inf` anywhere
- a single-tick jump above ~2 000 000°/tick — that is radians or a desync, not input
- an unrecognised `label`

Warnings (the behaviour is unusual, the file is fine):

- steps faster than 1 200°/s

Fast motion is a warning, deliberately. A snapped control signal genuinely
crosses tens of degrees in one tick, and an ingest layer that rejects it would
reject the exact signal the tool exists to measure.

## Units, because this is the usual bug

Degrees per second for velocity features, degrees for angles, seconds for time.
If your pipeline stores radians, multiply by `180 / pi` before ingest, not after.
`tracelab ingest` will tell you which trace and which tick failed, never just
"invalid data".
