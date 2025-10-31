import math
import unittest
import networkx as nx

from smr_alignment.metric_alignment import MetricAlignmentOptimizer


TOLERANCE = 1e-8


class TestMetricAlignmentOptimizer(unittest.TestCase):
    def setUp(self):
        """
        Build a tiny directed triangle A->B->C and A->C with two metrics:
          - m_good respects the similarity triangle inequality exactly:
                s_AC = s_AB * s_BC = 0.9 * 0.9 = 0.81
          - m_bad violates it strongly:
                s_AC << s_AB * s_BC
        Optimizer should push weight to m_good ≈ 1.
        """
        self.G = nx.DiGraph()
        # A -> B
        self.G.add_edge("A", "B", metric_scores={"m_good": 0.9, "m_bad": 0.9})
        self.G["A"]["B"]["weight"] = max(self.G["A"]["B"]["metric_scores"].values())

        # B -> C
        self.G.add_edge("B", "C", metric_scores={"m_good": 0.9, "m_bad": 0.9})
        self.G["B"]["C"]["weight"] = max(self.G["B"]["C"]["metric_scores"].values())

        # A -> C (direct)
        self.G.add_edge("A", "C", metric_scores={"m_good": 0.81, "m_bad": 0.2})
        self.G["A"]["C"]["weight"] = max(self.G["A"]["C"]["metric_scores"].values())

        self.mao = MetricAlignmentOptimizer(self.G)

    def test_optimize_weights_prefers_triangle_consistent_metric(self):
        w = self.mao.optimize_weights(eps=1e-8, entropy_lambda=0.0)
        # keys present
        self.assertIn("m_good", w)
        self.assertIn("m_bad", w)
        # nonnegative and sum to 1
        self.assertGreaterEqual(w["m_good"], -TOLERANCE)
        self.assertGreaterEqual(w["m_bad"], -TOLERANCE)
        self.assertAlmostEqual(w["m_good"] + w["m_bad"], 1.0, places=7)

        # should strongly prefer m_good
        self.assertGreater(w["m_good"], 0.95)
        self.assertLess(w["m_bad"], 0.05)

    def test_collect_pair_logs_sets_attribute(self):
        pair_logs = self.mao._collect_pair_logs(eps=1e-8, attr="log_scores_by_metric")
        # attribute set on all edges and finite
        for _, _, d in self.G.edges(data=True):
            self.assertIn("log_scores_by_metric", d)
            for val in d["log_scores_by_metric"].values():
                self.assertTrue(math.isfinite(val))

        # pair_logs keys match edges
        self.assertEqual(set(pair_logs.keys()), {("A", "B"), ("B", "C"), ("A", "C")})

    def test_alpha_vectors_nonempty_and_correct_width(self):
        pair_logs = self.mao._collect_pair_logs(eps=1e-8)
        metrics = self.mao.metrics_
        self.assertIsNotNone(metrics)
        A = self.mao._alpha_vectors(pair_logs, metrics)
        # shape: (num_triplets*orientations, num_metrics)
        self.assertEqual(A.shape[1], len(metrics))
        self.assertGreater(A.shape[0], 0)

    def test_apply_unified_scores_sets_expected_fields_and_weight_policy(self):
        w = {"m_good": 0.97, "m_bad": 0.03}  # simulate learned weights
        self.mao.apply_unified_scores(weights=w, replace_weight=True)

        # Check one edge thoroughly
        d = self.G["A"]["C"]
        self.assertIn("unified_log_cost", d)
        self.assertIn("unified_similarity", d)
        self.assertIn("metric_weights", d)
        self.assertIn("metric_scores_weighted", d)

        # per-edge subset of weights only includes present metrics
        self.assertEqual(set(d["metric_weights"].keys()), {"m_good", "m_bad"})
        self.assertAlmostEqual(d["metric_weights"]["m_good"], 0.97, places=7)
        self.assertAlmostEqual(d["metric_weights"]["m_bad"], 0.03, places=7)

        # expected unified score/product: prod s_m^{w_m}
        s_good = 0.81
        s_bad = 0.2
        s_hat_expected = (s_good ** 0.97) * (s_bad ** 0.03)
        self.assertAlmostEqual(d["unified_similarity"], s_hat_expected, places=10)

        # expected log cost
        c_expected = 0.97 * (-math.log(s_good)) + 0.03 * (-math.log(s_bad))
        self.assertAlmostEqual(d["unified_log_cost"], c_expected, places=10)

        # metric_scores_weighted holds s_m^{w_m}
        self.assertIn("m_good", d["metric_scores_weighted"])
        self.assertIn("m_bad", d["metric_scores_weighted"])
        self.assertAlmostEqual(d["metric_scores_weighted"]["m_good"], s_good ** 0.97, places=10)
        self.assertAlmostEqual(d["metric_scores_weighted"]["m_bad"], s_bad ** 0.03, places=10)

        # working weight policy: max of weighted metric scores
        expected_operational = max(s_good ** 0.97, s_bad ** 0.03)
        self.assertAlmostEqual(d["weight"], expected_operational, places=10)

    def test_no_metrics_returns_empty_weights_and_no_crash_on_apply(self):
        G2 = nx.DiGraph()
        G2.add_edge("X", "Y", weight=0.5)  # no metric_scores
        mao2 = MetricAlignmentOptimizer(G2)
        w = mao2.optimize_weights(eps=1e-8)
        self.assertEqual(w, {})  # nothing to learn

        # apply should no-op gracefully
        mao2.apply_unified_scores(weights=w, replace_weight=True)
        # weight unchanged
        self.assertAlmostEqual(G2["X"]["Y"]["weight"], 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
