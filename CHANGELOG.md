# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `docs/SCHEMA.md`: the trace input contract, including what is an error and
  what is only a warning, and why fast motion must never be fatal.
- `ROADMAP.md`: release gates from proof of concept to 1.0.

## [0.2.0] — 2026-11-10

### Added
- **`ingest`**: JSONL, JSON and long-format CSV loaders with strict, trace-level
  validation. Errors name the trace and the tick; they never say "invalid data".
- **`model`**: versioned, portable model bundles saved as plain JSON. No pickle —
  a model file that can execute code is a supply-chain hole.
- **CLI**: `train`, `score`, `export`. `train` writes a bundle with provenance
  and an FPR-constrained threshold; `score` emits ranked output with the driving
  feature per trace.
- 12 new tests covering ingest round trips, rejection paths, warning-vs-error
  behaviour, and model save/load equivalence.

### Fixed
- Ingest rejected legitimate high-speed control motion. The velocity ceiling is
  now a warning at 1 200°/s and an error only above a physically impossible
  2 000 000°/tick, because a snap crossing 20° in one tick is the signal, not a
  defect.
- `to_jsonl` and `Bundle.save` failed on a missing parent directory.

## [0.1.0] — 2026-11-10

### Added
- Synthetic trace generator: `human`, `snap`, `smoothed`, with occlusion windows.
- 12 mechanism features from control angles plus a visibility flag.
- Logistic detector with class balancing, FPR-constrained operating point and
  percentile-bootstrap confidence intervals.
- Evaluation harness reporting precision at realistic prevalence, per-family
  recall, and an explicit evidence card for the borderline cases.
- Vision front-end: frame → control/target centroids → angles, 0.13° round-trip MAE.
- Self-contained HTML report (no JS, no CDN).
- 14 stdlib tests.
