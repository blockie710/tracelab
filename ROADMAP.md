# Roadmap: proof of concept → 1.0

Two tracks, released independently. The engine can hit 1.0 without the product,
and the product is worthless without the engine.

---

## Where it is now (v0.1)

| Capability | State |
|---|---|
| Synthetic trace generation | done, 3 classes, occlusion windows |
| 12 mechanism features | done, tested for the reason each one claims |
| Detector + FPR-constrained threshold | done, logistic, explainable |
| Evaluation with bootstrap CIs + prevalence arithmetic | done |
| Vision front-end (frames → angles) | done on synthetic frames, 0.13° MAE |
| HTML report | done |
| **Real data ingest** | **done (v0.2 work), JSONL + CSV with strict validation** |
| **Persisted model bundles** | **done (v0.2 work), versioned JSON, no pickle** |
| Train / score CLI | done |
| Model calibration against production data | **not started** — the real gap |
| Review-queue product | **not started** |

The honest summary: the mathematics is demonstrated and the plumbing now accepts
someone else's data. What has never happened is a single run against real
telemetry with real labels.

---

## v0.2 — "someone else can use it" (2–3 weeks)

- [x] JSONL/CSV ingest with loud validation
- [x] Versioned model bundles, `train` and `score` subcommands
- [ ] `docs/SCHEMA.md` published with a worked example dataset in `examples/`
- [ ] Deterministic mode throughout (`--seed` on every stochastic path)
- [ ] Performance budget documented: traces/second, memory per trace
- [ ] Packaging: `pip install tracelab`, wheel built in CI, `python -m tracelab`
- [ ] Deprecation policy in writing (what breaks, when, how it is announced)

**Gate:** an outsider clones the repo, points it at their own JSONL, trains, scores,
and gets a sane number without reading the source.

## v0.5 — "trustworthy numbers" (6–8 weeks)

- [ ] Calibration module: map scores to probabilities on a labelled holdout
- [ ] Reliability diagram + ECE in the report; a probability you cannot check is
      a decoration
- [ ] Cross-validation and temporal-split helpers
- [ ] Per-feature ablation table (which features actually carry the signal, and
      which are decoration)
- [ ] Adversarial suite: a `smoothed` generator with tunable realism, scored
      against the detector, so regression is measurable
- [ ] Reference datasets with provenance and licence, published separately from
      the code

**Gate:** reported metrics survive a reviewer trying to break them — different
seed, different split, held-out family.

## v1.0 — "a product someone can depend on" (3–4 months)

- [ ] Stable public API with semantic versioning and a documented compatibility
      guarantee
- [ ] Model registry conventions: naming, versioning, retention, rollback
- [ ] Batch scoring at scale: streaming JSONL, bounded memory, resumable
- [ ] Review-queue surface: ranked list, evidence card per item, export to CSV,
      no web stack required
- [ ] Operability: structured logs, metrics endpoint, health check, one-command
      container
- [ ] Security posture documented: threat model, dependency policy, disclosure
      process
- [ ] Two independent reproductions of the headline numbers by someone who is
      not the author

**Gate:** a second organisation runs it in anger for a month and reports the same
conclusions you do.

---

## Deliberately not on the roadmap

- **Automatic enforcement.** The precision arithmetic at realistic prevalence
  makes automated action indefensible without a human in the loop. If a
  deployment wants auto-action, that is a product decision with a legal
  signature attached, not a library feature.
- **Client-side collection.** Out of scope by design. Everything here works from
  data a service already holds.
- **A hosted service.** v1.0 is a library and a batch tool. A hosted product is a
  different company shape: multi-tenancy, data processing agreements, uptime
  commitments. Do not smuggle it in as a feature.

---

## Release mechanics

- Tagged releases, `CHANGELOG.md` in Keep-a-Changelog form
- Wheels for each supported Python, tested on 3.9 / 3.12 / 3.13 / 3.14
- CI must pass on the release commit; a release cut from a red build is not a
  release
- Every model bundle records the git SHA and dataset hash that produced it
