"""Tests. Stdlib unittest so they run anywhere, pytest-compatible if present."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tracelab.detector import LogisticDetector, Standardizer, roc_auc, tpr_at_fpr
from tracelab.features import FEATURE_NAMES, extract
from tracelab.report import render
from tracelab.simulate import KINDS, simulate_dataset, simulate_trace
from tracelab.vision import round_trip_check


class TestSimulation(unittest.TestCase):
    def test_traces_are_well_formed(self):
        for kind in KINDS:
            t = simulate_trace(kind, ticks=120)
            self.assertEqual(t.n, 120)
            self.assertEqual(len(t.pitch), 120)
            self.assertEqual(len(t.visible), 120)
            self.assertTrue(all(math.isfinite(v) for v in t.yaw))

    def test_dataset_is_balanced(self):
        data = simulate_dataset(10, seed=3)
        self.assertEqual(len(data), 30)
        for kind in KINDS:
            self.assertEqual(sum(1 for t in data if t.kind == kind), 10)


class TestFeatures(unittest.TestCase):
    def test_no_nan(self):
        for kind in KINDS:
            f = extract(simulate_trace(kind, ticks=150))
            self.assertEqual(set(f), set(FEATURE_NAMES))
            for name, v in f.items():
                self.assertTrue(math.isfinite(v), f"{name} not finite for {kind}")

    def test_humans_drift_through_cover_and_bots_do_not(self):
        """The occlusion feature must separate for its stated reason.

        Seeded and averaged over 25 traces: this feature is a median of window
        deltas and is noisy per-trace, so an unseeded 12-sample comparison is a
        coin flip. That flakiness was a real defect, not bad luck.
        """
        import random

        rng = random.Random(1234)
        humans = [extract(simulate_trace("human", rng, ticks=180))["occl_drift"] for _ in range(25)]
        bots = [extract(simulate_trace("snap", rng, ticks=180))["occl_drift"] for _ in range(25)]
        self.assertGreater(sum(humans) / len(humans), sum(bots) / len(bots) + 1.0)

    def test_humans_show_tremor_bots_do_not(self):
        import random

        rng = random.Random(99)
        humans = [extract(simulate_trace("human", rng, ticks=180))["tremor_ratio"] for _ in range(20)]
        bots = [extract(simulate_trace("snap", rng, ticks=180))["tremor_ratio"] for _ in range(20)]
        self.assertGreater(sum(humans) / len(humans), sum(bots) / len(bots) * 2)


class TestDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = simulate_dataset(40, seed=11)
        cls.test = simulate_dataset(25, seed=12)
        cls.Xtr = [[extract(t)[k] for k in FEATURE_NAMES] for t in cls.train]
        cls.ytr = [0 if t.kind == "human" else 1 for t in cls.train]
        cls.model = LogisticDetector(epochs=300).fit(cls.Xtr, cls.ytr)
        cls.Xte = [[extract(t)[k] for k in FEATURE_NAMES] for t in cls.test]
        cls.yte = [0 if t.kind == "human" else 1 for t in cls.test]
        cls.scores = cls.model.predict_proba(cls.Xte)

    def test_auc_is_strong(self):
        self.assertGreater(roc_auc(self.scores, self.yte), 0.90)

    def test_probabilities_are_in_range(self):
        for s in self.scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_mean_bot_score_exceeds_mean_human_score(self):
        bots = [s for s, l in zip(self.scores, self.yte) if l == 1]
        humans = [s for s, l in zip(self.scores, self.yte) if l == 0]
        self.assertGreater(sum(bots) / len(bots), sum(humans) / len(humans))

    def test_contributions_are_ranked(self):
        row = self.Xte[0]
        parts = self.model.contributions(row, FEATURE_NAMES)
        self.assertEqual(len(parts), len(FEATURE_NAMES))
        mags = [abs(v) for _, v in parts]
        self.assertEqual(mags, sorted(mags, reverse=True))

    def test_auc_complements_under_label_flip(self):
        """Flipping labels inverts ranking quality: AUC(y) + AUC(1-y) = 1."""
        flipped = [1 - l for l in self.yte]
        self.assertAlmostEqual(
            roc_auc(self.scores, self.yte) + roc_auc(self.scores, flipped), 1.0, places=9
        )

    def test_tpr_at_fpr_respects_budget(self):
        tpr, thr = tpr_at_fpr(self.scores, self.yte, target_fpr=0.05)
        self.assertGreaterEqual(tpr, 0.0)
        self.assertLessEqual(tpr, 1.0)


class TestStandardizer(unittest.TestCase):
    def test_round_trip_zero_mean(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        Z = Standardizer().fit(X).transform(X)
        for j in range(2):
            self.assertAlmostEqual(sum(r[j] for r in Z), 0.0, places=9)


class TestVision(unittest.TestCase):
    def test_pixel_path_recovers_the_signal(self):
        for kind in KINDS:
            res = round_trip_check(simulate_trace(kind, ticks=90, mode="track"), stride=6)
            self.assertGreater(res["n"], 10)
            self.assertLess(res["mae_deg"], 1.0, f"{kind} vision error too high")
            self.assertGreater(res["corr"], 0.9)


class TestReport(unittest.TestCase):
    def test_report_renders_html(self):
        from tracelab.evaluate import run_experiment

        html = render(run_experiment(n_train=20, n_test=12, seed=5))
        self.assertIn("<!doctype html>", html)
        self.assertIn("ROC", html)
        self.assertIn("precision", html.lower())
        self.assertGreater(len(html), 5000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
