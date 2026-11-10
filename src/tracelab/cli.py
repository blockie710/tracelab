"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from .evaluate import run_experiment
from .report import write_report
from .simulate import simulate_trace
from .vision import round_trip_check, save_sample_frames


def cmd_demo(args) -> int:
    result = run_experiment(seed=args.seed)
    write_report(result, args.out)
    m = result["metrics"]
    lo, hi = m["auc_ci95"]
    print("tracelab — kinematic trace analysis")
    print("=" * 56)
    print(f"  test traces     {m['n_test']}")
    print(f"  AUC             {m['auc']:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
    print(f"  recall @1% FPR  {m['tpr_at_1pct_fpr']*100:.1f}%")
    print(f"  realised FPR    {m['realised_fpr']*100:.2f}%  ({m['fp']} FP / {m['tn']+m['fp']} humans)")
    print(f"  raw snap      AUC {result['per_kind']['snap']['auc']:.4f}")
    print(f"  smoothed actor   AUC {result['per_kind']['smoothed']['auc']:.4f}")
    print("-" * 56)
    print("  precision of a ban, by real positive prevalence:")
    for row in result["prevalence_table"]:
        print(f"    {row['prevalence']*100:>4.1f}%  ->  {row['precision']*100:>5.1f}%")
    print("-" * 56)
    print(f"  wrote {args.out}")
    return 0


def cmd_eval(args) -> int:
    result = run_experiment(seed=args.seed, n_train=args.train, n_test=args.test)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result["metrics"], indent=2))
    return 0


def cmd_vision(args) -> int:
    import random

    rng = random.Random(args.seed)
    print("pixel -> degrees round trip (stride %d)" % args.stride)
    print("=" * 56)
    for kind in ("human", "snap", "smoothed"):
        trace = simulate_trace(kind, rng, ticks=120, mode="track")
        res = round_trip_check(trace, stride=args.stride)
        print(
            f"  {kind:18s} n={res['n']:3d}  MAE={res['mae_deg']:.2f} deg  "
            f"corr={res['corr']:.3f}"
        )
    if args.save_frames:
        trace = simulate_trace("snap", rng, ticks=60, mode="track")
        paths = save_sample_frames(trace, [0, 20, 40], args.save_frames)
        print(f"  wrote {len(paths)} frames to {args.save_frames}")
    return 0


def cmd_simulate(args) -> int:
    import random

    rng = random.Random(args.seed)
    trace = simulate_trace(args.kind, rng, ticks=args.ticks, mode=args.mode)
    print(json.dumps(trace.as_dict(), indent=None)[:2000])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tracelab", description=__doc__)
    p.add_argument("--version", action="version", version="tracelab 0.1.0")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the benchmark and write the HTML report")
    d.add_argument("--out", default="demo/index.html")
    d.add_argument("--seed", type=int, default=20261110)
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("eval", help="print metrics as JSON")
    e.add_argument("--seed", type=int, default=20261110)
    e.add_argument("--train", type=int, default=220)
    e.add_argument("--test", type=int, default=110)
    e.add_argument("--json")
    e.set_defaults(func=cmd_eval)

    v = sub.add_parser("vision", help="run the pixel->degrees round trip check")
    v.add_argument("--seed", type=int, default=4)
    v.add_argument("--stride", type=int, default=3)
    v.add_argument("--save-frames")
    v.set_defaults(func=cmd_vision)

    s = sub.add_parser("simulate", help="dump one synthetic trace as JSON")
    s.add_argument("--kind", default="human", choices=["human", "snap", "smoothed"])
    s.add_argument("--mode", default="track", choices=["track", "flick"])
    s.add_argument("--ticks", type=int, default=180)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_simulate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
