"""Command line entry point."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from .detector import LogisticDetector
from .evaluate import run_experiment
from .features import FEATURE_NAMES, extract
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


def cmd_export(args) -> int:
    from .ingest import to_jsonl
    from .simulate import simulate_dataset

    data = simulate_dataset(args.per_kind, seed=args.seed, ticks=args.ticks)
    for i, t in enumerate(data):
        setattr(t, "trace_id", f"{t.kind}-{i:04d}")
    to_jsonl(data, args.out)
    print(f"wrote {len(data)} traces to {args.out}")
    return 0


def cmd_train(args) -> int:
    from .detector import tpr_at_fpr
    from .ingest import load
    from .model import train_bundle

    traces = load(args.data)
    labelled = [t for t in traces if t.kind != "unknown"]
    if len(labelled) < 20:
        print(f"need at least 20 labelled traces, got {len(labelled)}", file=sys.stderr)
        return 2
    pos = sum(1 for t in labelled if t.kind != "human")
    neg = len(labelled) - pos
    if not pos or not neg:
        print("need both positive and negative labels", file=sys.stderr)
        return 2

    X = [[extract(t)[k] for k in FEATURE_NAMES] for t in labelled]
    y = [0 if t.kind == "human" else 1 for t in labelled]
    model = LogisticDetector(l2=args.l2, epochs=args.epochs).fit(X, y)
    scores = model.predict_proba(X)
    tpr, thr = tpr_at_fpr(scores, y, target_fpr=args.target_fpr)

    bundle = train_bundle(
        model,
        name=args.name,
        metrics={
            "tpr_at_target_fpr": round(tpr, 4),
            "target_fpr": args.target_fpr,
            "n_train": len(labelled),
            "n_pos": pos,
            "n_neg": neg,
            "in_sample": True,
        },
        training={"source": args.data, "l2": args.l2, "epochs": args.epochs},
        threshold=thr,
        target_fpr=args.target_fpr,
    )
    bundle.save(args.out)
    print(f"trained on {len(labelled)} traces ({pos} pos / {neg} neg)")
    print(f"  TPR @ {args.target_fpr:.1%} FPR : {tpr:.3f}  (threshold {thr:.3f})")
    print(f"  wrote {args.out}")
    print("  note: metrics are in-sample; use `tracelab eval` on held-out data")
    return 0


def cmd_score(args) -> int:
    from .ingest import load
    from .model import Bundle

    bundle = Bundle.load(args.model)
    traces = load(args.data)
    rows = []
    for t in traces:
        s = bundle.score_trace(t)
        rows.append(
            {
                "trace_id": getattr(t, "trace_id", "?"),
                "reference_label": t.kind,
                "score": round(s, 6),
                "decision": bundle.decision(s),
                "top_feature": bundle.explain(t, top=1)[0][0],
            }
        )
    rows.sort(key=lambda r: -r["score"])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            if args.out.endswith(".csv"):
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            else:
                json.dump(rows, fh, indent=2)
        print(f"wrote {len(rows)} scored traces to {args.out}")

    flagged = sum(1 for r in rows if r["decision"] == "flag")
    print(f"{len(rows)} traces scored, {flagged} flagged at threshold {bundle.card.threshold:.3f}")
    for r in rows[: args.top]:
        print(
            f"  {r['trace_id']:32s} {r['score']:.3f}  {r['decision']:5s} "
            f"({r['reference_label']}, driven by {r['top_feature']})"
        )
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

    x = sub.add_parser("export", help="write a synthetic dataset as JSONL")
    x.add_argument("--out", default="data/simulated.jsonl")
    x.add_argument("--per-kind", type=int, default=200, dest="per_kind")
    x.add_argument("--ticks", type=int, default=180)
    x.add_argument("--seed", type=int, default=7)
    x.set_defaults(func=cmd_export)

    t = sub.add_parser("train", help="fit a model from labelled traces and save a bundle")
    t.add_argument("--data", required=True, help=".jsonl, .json or .csv of traces")
    t.add_argument("--out", default="models/tracelab.json")
    t.add_argument("--name", default="tracelab-default")
    t.add_argument("--l2", type=float, default=0.05)
    t.add_argument("--epochs", type=int, default=500)
    t.add_argument("--target-fpr", type=float, default=0.01, dest="target_fpr")
    t.set_defaults(func=cmd_train)

    sc = sub.add_parser("score", help="score traces with a saved bundle")
    sc.add_argument("--model", required=True)
    sc.add_argument("--data", required=True)
    sc.add_argument("--out", help="write .json or .csv")
    sc.add_argument("--top", type=int, default=5)
    sc.set_defaults(func=cmd_score)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
