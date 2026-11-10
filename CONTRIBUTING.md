# Contributing

## Before you write code

Open an issue describing the change and, more importantly, **what would make you
wrong**. This project's failure mode is a plausible feature that quietly makes
the numbers worse. A proposal that states its own falsification condition gets
reviewed faster than one that does not.

## Ground rules

1. **No new runtime dependencies.** The library ships with the standard library
   only. Optional extras may use `pillow`, and anything else needs a very good
   argument about supply chain and install weight.
2. **Every number gets an error bar.** If you report a metric, report its
   uncertainty. A point estimate from a few hundred traces is not a result.
3. **Report at realistic prevalence.** Balanced-test accuracy is not evidence.
   Any new metric must state the prevalence it assumes.
4. **Features must justify their mechanism.** If you add a feature, add a test
   that fails when the mechanism is removed — not just a test that the number is
   non-zero.
5. **Determinism.** Anything stochastic takes a seed. A test that passes on
   Tuesday and fails on Wednesday is worse than no test.

## Style

- Python 3.9+ syntax, `from __future__ import annotations` where you need it.
- Type hints on public functions.
- Docstrings explain *why*, not *what*. `# increment i` is noise.
- Keep functions small enough to read on one screen.

## Tests

```bash
make test        # 26 stdlib tests, no network, no fixtures on disk
make verify      # tests + benchmark + vision check
```

New behaviour needs a test that would fail without it. Bug fixes need a test
that reproduces the bug — and a comment saying what the bug was, because the
next person will try to "simplify" it back.

## Commits

Conventional-ish prefixes (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
The body explains the failure mode being addressed:

```
fix: ingest rejected legitimate fast motion

A 20 deg/tick ceiling failed on real snap traces. Fast motion is now a
warning; only physically impossible jumps (>2e6 deg/tick) are fatal.
Repro: tests/test_tracelab.py::TestIngest::test_fast_motion_warns_but_does_not_raise
```

## Reporting results

If you run this on your own data and get a number that disagrees with the
published one, that is the most valuable thing you can send. Include: dataset
shape, label provenance, tick rate, the split you used, and the exact command.
We will publish disagreements rather than average them away.

## Code of conduct

Be direct about the work and generous with the person. Critique the number, the
method, the evidence. Not the contributor.
