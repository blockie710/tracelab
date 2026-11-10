"""tracelab — kinematic trace analysis from server-side view angles.

Pure standard library. No numpy, no model weights, no client agent.
"""

from .simulate import Trace, TICK_HZ, simulate_trace
from .features import FEATURE_NAMES, extract, extract_many
from .detector import LogisticDetector, Standardizer, roc_auc, tpr_at_fpr

__version__ = "0.1.0"

__all__ = [
    "Trace",
    "TICK_HZ",
    "simulate_trace",
    "FEATURE_NAMES",
    "extract",
    "extract_many",
    "LogisticDetector",
    "Standardizer",
    "roc_auc",
    "tpr_at_fpr",
    "__version__",
]
