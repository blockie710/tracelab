# tracelab

Kinematic analytics for high-rate input telemetry: feature extraction, detector
evaluation, and reporting, in pure Python with no runtime dependencies.

It answers one question well: given a two-axis control signal sampled at 60 Hz,
**is this trace under organic control or scripted control?** — and it answers it
with an operating point chosen by a false-positive budget rather than by
accuracy on a balanced test set.

```
AUC (scripted vs organic)   0.9905   95% CI 0.9819–0.9966
recall @ 1% FPR             75.5%    realised FPR 0.91%
raw snap  AUC               1.0000
smoothed  AUC               0.9811   recall 50.9% at that threshold
pixel→angle round trip      0.13°    corr 0.998
```

Reproduce in about ten seconds:

```bash
make verify            # 14 tests + benchmark + vision check
make demo              # writes demo/index.html
```

## Why a control signal

Any real-time system that moves a cursor, a turret, or a camera already records
where it was pointed, 60+ times a second. That signal is free, already stored,
and requires nothing instrumented on the client — no privileged agent, no
elevated component that becomes an attack surface in its own right.

It is also hard to forge convincingly. A scripted actor can imitate a *mean*
easily. Imitating an entire *distribution* — reversal structure, tremor
periodicity, behaviour under occlusion — is a much harder problem, and that gap
is what this library measures.

## What it measures

Twelve features, all derivable from control angles plus a visibility flag. They
encode **mechanism**, not skill:

| Group | Features | What it encodes |
|---|---|---|
| Acquisition | `acquire_time_s`, `peak_speed_dps`, `peak_accel_dps2` | A snap arrives in one ballistic move; organic control arrives in a curve with corrections |
| Steadiness | `err_median_vis`, `err_std_vis`, `reversal_rate`, `err_autocorr1` | Organic control oscillates around the target; snaps converge and stop |
| Physiology | `tremor_ratio` | 9–11 Hz physiological tremor, measured on control velocity while on target |
| Occlusion | `occl_drift`, `cover_track_frac` | An operator loses a hidden target; a state-reading actor does not |
| Shape | `speed_kurtosis`, `stationary_frac`, `overshoot_ratio` | Bimodal "still, then huge" velocity is characteristic of snapping |

Three reference classes are simulated: `human`, `snap`, and `smoothed` (a slower
snap with injected tremor — the realistic hard case).

## The honest part

**Balanced accuracy is a vanity metric.** At a 1% false-positive rate and a 2%
positive prevalence, two in five flagged cases would be wrong:

| positive prevalence | precision of a flag |
|---|---|
| 0.5% | 29.4% |
| 1% | 45.6% |
| 2% | 62.9% |
| 5% | 81.4% |

That table is why the tool outputs a **ranked review queue with evidence cards**
rather than verdicts. Every score ships with its top feature contributions so a
human can audit the reason, not just the number.

**What this is not:**

- **Not validated on production data.** The traces are synthetic, generated from
  a spring-damper limb model. The numbers are real, but they measure the model.
- **Not full coverage.** `cover_track_frac` catches state-reading behaviour
  expressed through the control signal. An actor who uses out-of-band
  information without changing how the control signal moves is not caught here.
- **Not robust to a reader of this repository.** Everything is public by design;
  a determined adversary can tune against these specific features.

## Layout

```
src/tracelab/
  simulate.py   synthetic traces: organic limb vs snap, with occlusion windows
  features.py   12 mechanism features from control angles
  detector.py   logistic regression, AUC, FPR-constrained threshold, bootstrap CI
  evaluate.py   train/test split, per-family metrics, prevalence arithmetic
  vision.py     frames → control/target → angles → same features
  report.py     self-contained HTML report
  cli.py        demo | eval | vision | simulate
tests/          14 stdlib tests (unittest; pytest-compatible)
```

## Usage

```python
from tracelab import simulate_trace, extract

trace = simulate_trace("snap", ticks=180)
print({k: round(v, 3) for k, v in extract(trace).items()})
```

```bash
tracelab demo --out demo/index.html    # benchmark + dashboard
tracelab eval --json metrics.json      # metrics as JSON
tracelab vision                        # pixel→angle round-trip check
tracelab simulate --kind human         # dump one trace
```

## Requirements

Python 3.9+. No runtime dependencies. `pillow` only for the optional vision
module.

## License

MIT.
