"""Persisted, versioned detector bundles.

A trained model that only exists inside a Python session is a demo. Anything
shipped needs to be written to disk, versioned, and re-loadable years later by
someone who is not you — with enough metadata to answer "what was this thing,
what did it score, and when was it calibrated".

Format is plain JSON. No pickle: a pickle is a security hole the moment a model
file is shared, and an opaque binary is unacceptable for a system whose whole
claim is auditability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

from .detector import LogisticDetector, _sigmoid
from .features import FEATURE_NAMES, extract
from .simulate import Trace

FORMAT_VERSION = 1


@dataclass
class ModelCard:
    """Provenance. Shipping a score without this is shipping an opinion."""

    name: str
    created_utc: str
    features: List[str]
    training: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    threshold: float = 0.5
    target_fpr: float = 0.01
    notes: str = ""
    format_version: int = FORMAT_VERSION

    def to_dict(self) -> Dict:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "created_utc": self.created_utc,
            "features": self.features,
            "training": self.training,
            "metrics": self.metrics,
            "threshold": self.threshold,
            "target_fpr": self.target_fpr,
            "notes": self.notes,
        }


@dataclass
class Bundle:
    """A model you can hand to someone else."""

    card: ModelCard
    weights: List[float]
    bias: float
    mu: List[float]
    sd: List[float]

    # --- scoring ---------------------------------------------------------

    def score_trace(self, trace: Trace) -> float:
        row = [extract(trace)[k] for k in self.card.features]
        return self.score_row(row)

    def score_row(self, row: Sequence[float]) -> float:
        if len(row) != len(self.card.features):
            raise ValueError(
                f"expected {len(self.card.features)} features, got {len(row)}"
            )
        z = self.bias
        for w, m, s, x in zip(self.weights, self.mu, self.sd, row):
            z += w * ((x - m) / s)
        return _sigmoid(z)

    def decision(self, score: float) -> str:
        if score >= self.card.threshold:
            return "flag"
        return "pass"

    def explain(self, trace: Trace, top: int = 5) -> List[Tuple[str, float]]:
        row = [extract(trace)[k] for k in self.card.features]
        parts = [
            (name, w * ((x - m) / s))
            for name, w, m, s, x in zip(self.card.features, self.weights, self.mu, self.sd, row)
        ]
        parts.sort(key=lambda kv: -abs(kv[1]))
        return parts[:top]

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "card": self.card.to_dict(),
            "model": {"weights": self.weights, "bias": self.bias, "mu": self.mu, "sd": self.sd},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Bundle":
        version = data.get("card", {}).get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"model format_version {version!r} not supported (this build reads {FORMAT_VERSION})"
            )
        card = ModelCard(**{k: v for k, v in data["card"].items() if k != "format_version"})
        m = data["model"]
        if len(m["weights"]) != len(card.features):
            raise ValueError("model weight count does not match feature list")
        return cls(card=card, weights=m["weights"], bias=m["bias"], mu=m["mu"], sd=m["sd"])

    def save(self, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Bundle":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def train_bundle(
    detector: LogisticDetector,
    *,
    name: str,
    metrics: Dict | None = None,
    training: Dict | None = None,
    threshold: float = 0.5,
    target_fpr: float = 0.01,
    notes: str = "",
) -> Bundle:
    """Wrap a fitted LogisticDetector into a portable bundle."""
    if not detector.w:
        raise ValueError("detector is not fitted")
    features = list(FEATURE_NAMES)
    if len(detector.w) != len(features):
        raise ValueError(
            f"detector has {len(detector.w)} weights but {len(features)} features — "
            "was it trained on a different feature set?"
        )
    card = ModelCard(
        name=name,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        features=features,
        training=training or {},
        metrics=metrics or {},
        threshold=threshold,
        target_fpr=target_fpr,
        notes=notes,
    )
    return Bundle(
        card=card,
        weights=list(detector.w),
        bias=float(detector.b),
        mu=list(detector.scaler.mu),
        sd=list(detector.scaler.sd),
    )
