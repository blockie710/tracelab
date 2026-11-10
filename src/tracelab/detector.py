"""A small, fully explainable detector.

Deliberately a logistic regression rather than a black box. Two reasons, both
practical:

1. A decision that affects a person has to survive a challenge. "Feature X was
   6.2 sd above the reference population" is defensible; "the model said 0.97"
   is not.
2. It trains in under a second on a laptop, so the whole pipeline is
   reproducible by anyone who clones the repo.

A gradient-boosted or sequence model is the right production choice; this is
the honest baseline that proves the features carry signal.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


class Standardizer:
    def __init__(self) -> None:
        self.mu: List[float] = []
        self.sd: List[float] = []

    def fit(self, X: Sequence[Sequence[float]]) -> "Standardizer":
        if not X:
            return self
        d = len(X[0])
        self.mu = [0.0] * d
        self.sd = [0.0] * d
        for row in X:
            for j in range(d):
                self.mu[j] += row[j]
        self.mu = [m / len(X) for m in self.mu]
        for row in X:
            for j in range(d):
                self.sd[j] += (row[j] - self.mu[j]) ** 2
        self.sd = [math.sqrt(v / max(len(X) - 1, 1)) or 1.0 for v in self.sd]
        return self

    def transform(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        return [[(row[j] - self.mu[j]) / self.sd[j] for j in range(len(row))] for row in X]


def _sigmoid(z: float) -> float:
    if z < -35:
        return 0.0
    if z > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class LogisticDetector:
    """L2-regularised logistic regression, full-batch gradient descent."""

    def __init__(self, l2: float = 0.05, lr: float = 0.15, epochs: int = 600) -> None:
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.w: List[float] = []
        self.b: float = 0.0
        self.scaler = Standardizer()
        self.loss_curve: List[float] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "LogisticDetector":
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        Z = self.scaler.fit(X).transform(X)
        d = len(Z[0])
        self.w = [0.0] * d
        self.b = 0.0
        n = len(Z)
        pos = sum(y)
        neg = n - pos
        # Balance the classes so the minority (positives) is not drowned out.
        wpos = n / (2.0 * pos) if pos else 1.0
        wneg = n / (2.0 * neg) if neg else 1.0

        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            loss = 0.0
            for row, label in zip(Z, y):
                z = self.b + sum(wi * xi for wi, xi in zip(self.w, row))
                p = _sigmoid(z)
                cw = wpos if label == 1 else wneg
                err = (p - label) * cw
                for j in range(d):
                    gw[j] += err * row[j]
                gb += err
                pp = min(max(p, 1e-9), 1 - 1e-9)
                loss -= cw * (label * math.log(pp) + (1 - label) * math.log(1 - pp))
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * gb / n
            self.loss_curve.append(loss / n)
        return self

    def decision(self, row: Sequence[float]) -> float:
        z = self.scaler.transform([row])[0]
        return self.b + sum(wi * xi for wi, xi in zip(self.w, z))

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        return [_sigmoid(self.decision(row)) for row in X]

    def contributions(self, row: Sequence[float], names: Sequence[str]) -> List[Tuple[str, float]]:
        """Per-feature contribution to the logit, for the evidence card."""
        z = self.scaler.transform([row])[0]
        parts = [(names[j], self.w[j] * z[j]) for j in range(len(names))]
        parts.sort(key=lambda kv: -abs(kv[1]))
        return parts


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUC with proper tie handling."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        rank_sum += avg_rank * sum(lbl for _, lbl in pairs[i:j + 1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _thresholds(scores: Sequence[float], steps: int = 400) -> List[float]:
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]


def roc_points(scores: Sequence[float], labels: Sequence[int], steps: int = 400):
    """(fpr, tpr) pairs, threshold descending."""
    pts = []
    n_pos = sum(labels) or 1
    n_neg = len(labels) - sum(labels) or 1
    for t in _thresholds(scores, steps):
        tp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
        pts.append((fp / n_neg, tp / n_pos))
    return pts


def tpr_at_fpr(scores: Sequence[float], labels: Sequence[int], target_fpr: float = 0.01):
    """Best TPR achievable while keeping false positives at or below target."""
    n_pos = sum(labels) or 1
    n_neg = len(labels) - sum(labels) or 1
    best_tpr, best_thr = 0.0, max(scores)
    for t in _thresholds(scores):
        fp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
        if fp / n_neg <= target_fpr:
            tp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
            if tp / n_pos >= best_tpr:
                best_tpr, best_thr = tp / n_pos, t
    return best_tpr, best_thr


def bootstrap_ci(values: Sequence[float], stat, n_boot: int = 300, alpha: float = 0.05, seed: int = 7):
    """Percentile bootstrap. Headline numbers without error bars are theatre."""
    rng = random.Random(seed)
    vals = list(values)
    out = []
    for _ in range(n_boot):
        sample = [rng.choice(vals) for _ in vals]
        out.append(stat(sample))
    out.sort()
    lo = out[int(alpha / 2 * len(out))]
    hi = out[int((1 - alpha / 2) * len(out)) - 1]
    return lo, hi
