"""Evaluation harness.

Three things this does that a marketing benchmark does not:

1. Trains and tests on disjoint samples.
2. Reports bootstrapped confidence intervals, because a point estimate from a
   few hundred traces is not a result.
3. Reports precision at *realistic prevalence*, not at the balanced prevalence
   of the test set. Detection systems look excellent on balanced
   data and terrible on a live population, and that gap is where trust dies.
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List, Sequence

from .detector import (
    LogisticDetector,
    bootstrap_ci,
    roc_auc,
    roc_points,
    tpr_at_fpr,
)
from .features import FEATURE_NAMES, extract
from .simulate import KINDS, simulate_dataset


def _scores_and_labels(traces, model) -> tuple:
    rows = [[extract(t)[k] for k in FEATURE_NAMES] for t in traces]
    labels = [0 if t.kind == "human" else 1 for t in traces]
    return model.predict_proba(rows), labels, rows


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """Positive predictive value at a given real-world positive rate."""
    tp = tpr * prevalence
    fp = fpr * (1.0 - prevalence)
    return tp / (tp + fp) if (tp + fp) else 0.0


def run_experiment(
    n_train: int = 220,
    n_test: int = 110,
    seed: int = 20261110,
    l2: float = 0.05,
) -> Dict:
    train = simulate_dataset(n_train, seed=seed)
    test = simulate_dataset(n_test, seed=seed + 1)

    Xtr = [[extract(t)[k] for k in FEATURE_NAMES] for t in train]
    ytr = [0 if t.kind == "human" else 1 for t in train]
    model = LogisticDetector(l2=l2, epochs=500).fit(Xtr, ytr)

    scores, labels, rows = _scores_and_labels(test, model)

    auc = roc_auc(scores, labels)
    tpr, thr = tpr_at_fpr(scores, labels, target_fpr=0.01)

    # Realised confusion at that operating point.
    tp = sum(1 for s, l in zip(scores, labels) if s >= thr and l == 1)
    fp = sum(1 for s, l in zip(scores, labels) if s >= thr and l == 0)
    fn = sum(1 for s, l in zip(scores, labels) if s < thr and l == 1)
    tn = sum(1 for s, l in zip(scores, labels) if s < thr and l == 0)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    realised_fpr = fp / n_neg if n_neg else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    # Bootstrap CIs over the paired (score, label) test rows.
    pairs = list(zip(scores, labels))
    lo, hi = bootstrap_ci(
        pairs, lambda s: roc_auc([a for a, _ in s], [b for _, b in s])
    )

    # AUC restricted to one positive family vs negatives: the smoothed actor is
    # the hard case and deserves its own number.
    per_kind = {}
    for kind in KINDS:
        if kind == "human":
            continue
        mask = [i for i, t in enumerate(test) if t.kind in ("human", kind)]
        per_kind[kind] = {
            "auc": roc_auc([scores[i] for i in mask], [labels[i] for i in mask]),
            "recall_at_thr": (
                sum(1 for i in mask if test[i].kind == kind and scores[i] >= thr)
                / max(sum(1 for i in mask if test[i].kind == kind), 1)
            ),
        }

    # Per-class feature means, for the report and for sanity checking that each
    # feature separates for the reason it claims to.
    feat_means = {}
    for k in FEATURE_NAMES:
        feat_means[k] = {
            kind: statistics.mean(extract(t)[k] for t in test if t.kind == kind) for kind in KINDS
        }

    prevalence_table = []
    for prev in (0.005, 0.01, 0.02, 0.05, 0.10):
        prevalence_table.append(
            {
                "prevalence": prev,
                "precision": precision_at_prevalence(tpr, realised_fpr, prev),
                "flagged_per_10k": (tpr * prev + realised_fpr * (1 - prev)) * 10000,
            }
        )

    # Pick the most borderline cases: highest-scoring negative and
    # lowest-scoring positive. Those two decide whether the system is usable.
    neg_scores = [(s, i) for i, (s, l) in enumerate(zip(scores, labels)) if l == 0]
    pos_scores = [(s, i) for i, (s, l) in enumerate(zip(scores, labels)) if l == 1]
    strongest_negative = max(neg_scores, key=lambda p: p[0])
    weakest_positive = min(pos_scores, key=lambda p: p[0])

    evidence = {
        "flagged": {
            "kind": test[weakest_positive[1]].kind,
            "score": weakest_positive[0],
            "contributions": model.contributions(rows[weakest_positive[1]], FEATURE_NAMES)[:6],
        },
        "closest_human": {
            "kind": test[strongest_negative[1]].kind,
            "score": strongest_negative[0],
            "contributions": model.contributions(rows[strongest_negative[1]], FEATURE_NAMES)[:6],
        },
    }

    return {
        "config": {
            "n_train_per_kind": n_train,
            "n_test_per_kind": n_test,
            "seed": seed,
            "l2": l2,
            "tick_hz": 60,
            "features": FEATURE_NAMES,
        },
        "metrics": {
            "auc": auc,
            "auc_ci95": [lo, hi],
            "tpr_at_1pct_fpr": tpr,
            "threshold": thr,
            "realised_fpr": realised_fpr,
            "precision_balanced_test": precision,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_test": len(test),
        },
        "per_kind": per_kind,
        "feature_means": feat_means,
        "weights": dict(zip(FEATURE_NAMES, model.w)),
        "roi_of_roc": roc_points(scores, labels, steps=200),
        "score_distributions": {
            kind: sorted(s for s, t in zip(scores, test) if t.kind == kind) for kind in KINDS
        },
        "prevalence_table": prevalence_table,
        "evidence": evidence,
        "loss_curve": model.loss_curve[::10],
    }
