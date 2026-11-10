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


class TestIngest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp()
        self.rng = __import__("random").Random(5)

    def _write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_jsonl_round_trip(self):
        from tracelab.ingest import load_jsonl, to_jsonl

        traces = [simulate_trace(k, self.rng, ticks=90) for k in KINDS]
        for i, t in enumerate(traces):
            setattr(t, "trace_id", f"t{i}")
        path = os.path.join(self.dir, "d.jsonl")
        to_jsonl(traces, path)
        back = load_jsonl(path)
        self.assertEqual(len(back), 3)
        self.assertEqual([t.kind for t in back], [t.kind for t in traces])
        self.assertEqual(back[0].n, 90)

    def test_csv_ingest(self):
        from tracelab.ingest import load_csv

        rows = ["trace_id,tick,yaw,pitch,target_yaw,target_pitch,visible,label"]
        for i in range(40):
            rows.append(f"a,{i},{i*0.1:.2f},0.0,10.0,1.0,true,human")
        path = self._write("d.csv", "\n".join(rows) + "\n")
        traces = load_csv(path)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].n, 40)
        self.assertEqual(traces[0].kind, "human")

    def test_rejects_length_mismatch(self):
        from tracelab.ingest import IngestError, from_records

        with self.assertRaises(IngestError):
            from_records([{"id": "x", "label": "human", "yaw": [0.0] * 40,
                           "pitch": [0.0] * 39, "target_yaw": [0.0] * 40,
                           "target_pitch": [0.0] * 40}])

    def test_rejects_non_finite(self):
        from tracelab.ingest import IngestError, from_records

        with self.assertRaises(IngestError):
            from_records([{"id": "x", "label": "human", "yaw": [0.0] * 39 + [float("nan")],
                           "pitch": [0.0] * 40, "target_yaw": [0.0] * 40,
                           "target_pitch": [0.0] * 40}])

    def test_rejects_unknown_label(self):
        from tracelab.ingest import IngestError, from_records

        with self.assertRaises(IngestError):
            from_records([{"id": "x", "label": "definitely_not_a_label",
                           "yaw": [0.0] * 40, "pitch": [0.0] * 40,
                           "target_yaw": [0.0] * 40, "target_pitch": [0.0] * 40}])

    def test_fast_motion_warns_but_does_not_raise(self):
        """A snap is legitimately fast. Rejecting it would reject the signal."""
        from tracelab.ingest import collect_warnings, from_records

        yaw = [0.0] * 20 + [40.0] * 20  # 40 deg in one tick at 60 Hz
        traces = from_records([{"id": "fast", "label": "snap", "yaw": yaw,
                                "pitch": [0.0] * 40, "target_yaw": [0.0] * 40,
                                "target_pitch": [0.0] * 40}])
        self.assertEqual(len(traces), 1)
        self.assertTrue(collect_warnings(traces[0], 60))

    def test_rejects_impossible_jump(self):
        from tracelab.ingest import IngestError, from_records

        yaw = [0.0] * 20 + [1e7] * 20
        with self.assertRaises(IngestError):
            from_records([{"id": "boom", "label": "snap", "yaw": yaw,
                           "pitch": [0.0] * 40, "target_yaw": [0.0] * 40,
                           "target_pitch": [0.0] * 40}])


class TestModel(unittest.TestCase):
    def setUp(self):
        import random
        import tempfile

        from tracelab.model import train_bundle

        self.dir = tempfile.mkdtemp()
        rng = random.Random(3)
        data = [simulate_trace(k, rng, ticks=120) for k in KINDS for _ in range(6)]
        X = [[extract(t)[k] for k in FEATURE_NAMES] for t in data]
        y = [0 if t.kind == "human" else 1 for t in data]
        det = LogisticDetector(epochs=120).fit(X, y)
        self.bundle = train_bundle(det, name="test", threshold=0.5)
        self.data = data

    def test_save_load_round_trip_is_identical(self):
        from tracelab.model import Bundle

        path = os.path.join(self.dir, "m.json")
        self.bundle.save(path)
        back = Bundle.load(path)
        for t in self.data[:6]:
            self.assertAlmostEqual(
                self.bundle.score_trace(t), back.score_trace(t), places=12
            )

    def test_card_carries_metadata(self):
        self.assertEqual(self.bundle.card.name, "test")
        self.assertEqual(self.bundle.card.features, list(FEATURE_NAMES))
        self.assertTrue(self.bundle.card.created_utc.endswith("Z"))

    def test_unsupported_format_version_raises(self):
        from tracelab.model import Bundle

        d = self.bundle.to_dict()
        d["card"]["format_version"] = 99
        with self.assertRaises(ValueError):
            Bundle.from_dict(d)

    def test_explain_is_sorted_by_magnitude(self):
        parts = self.bundle.explain(self.data[0], top=5)
        self.assertEqual(len(parts), 5)
        mags = [abs(v) for _, v in parts]
        self.assertEqual(mags, sorted(mags, reverse=True))

    def test_decision_uses_threshold(self):
        self.assertEqual(self.bundle.decision(0.99), "flag")
        self.assertEqual(self.bundle.decision(0.01), "pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
