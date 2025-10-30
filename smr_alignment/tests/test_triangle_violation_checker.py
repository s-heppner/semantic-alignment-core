import math
import unittest

from smr.algorithm import SemanticMatchGraph
from smr_alignment.triangle_violation_checker import TriangleViolationChecker


TOLERANCE = 1e-12
EPSILON = 1e-8


class TestTriangleViolationChecker(unittest.TestCase):
    def make_graph_with_triangle_violation(self):
        """
        Build a tiny graph where:
          s(A,B)=0.9, s(B,C)=0.9, s(A,C)=0.5
        This violates s(A,C) >= s(A,B)*s(B,C) because 0.5 < 0.81.
        """
        G = SemanticMatchGraph()
        G.add_semantic_match("A", "B", 0.9)
        G.add_semantic_match("B", "C", 0.9)
        G.add_semantic_match("A", "C", 0.5)
        return G

    def make_graph_consistent_triangle(self):
        """
        Build a tiny graph where:
          s(A,B)=0.5, s(B,C)=0.5, s(A,C)=0.3
        This is consistent because 0.3 >= 0.25.
        """
        G = SemanticMatchGraph()
        G.add_semantic_match("A", "B", 0.5)
        G.add_semantic_match("B", "C", 0.5)
        G.add_semantic_match("A", "C", 0.3)
        return G

    def test_add_log_cost_clamps_both_sides(self):
        G = SemanticMatchGraph()
        G.add_semantic_match("u", "v", 0.0)         # will clamp to EPS
        G.add_semantic_match("v", "w", 1.0)         # will clamp to 1 - EPSILON
        G.add_semantic_match("w", "x", 0.5)         # in range

        tvc = TriangleViolationChecker(G)

        # u->v
        d = G["u"]["v"]
        self.assertAlmostEqual(d["original_similarity"], 0.0, places=12)
        self.assertAlmostEqual(d["log_cost"], -math.log(EPSILON), places=8)

        # v->w
        d = G["v"]["w"]
        self.assertAlmostEqual(d["original_similarity"], 1.0, places=12)
        self.assertAlmostEqual(d["log_cost"], -math.log(1 - EPSILON), places=8)

        # w->x
        d = G["w"]["x"]
        self.assertAlmostEqual(d["original_similarity"], 0.5, places=12)
        self.assertAlmostEqual(d["log_cost"], -math.log(0.5), places=12)

    def test_floyd_warshall_repairs_violation_and_sets_flags(self):
        G = self.make_graph_with_triangle_violation()
        tvc = TriangleViolationChecker(G)
        tvc.calculate_repaired_costs_floyd_warshall()

        # A->C should be repaired to product 0.9*0.9 = 0.81 in similarity
        # In cost space, repaired = -log(0.81) unless clamping changes it (it doesn't here)
        ac = G["A"]["C"]
        direct_cost = ac["log_cost"]
        repaired_cost = ac["log_cost_repaired"]

        self.assertLess(repaired_cost, direct_cost + TOLERANCE)  # improved
        expected_cost = -math.log(0.9) + -math.log(0.9)    # two-edge path cost
        self.assertAlmostEqual(repaired_cost, expected_cost, places=12)
        self.assertTrue(ac["violation"])

        # A->B and B->C should be unchanged (still optimal direct edges)
        self.assertFalse(G["A"]["B"]["violation"])
        self.assertFalse(G["B"]["C"]["violation"])
        self.assertAlmostEqual(G["A"]["B"]["log_cost_repaired"], G["A"]["B"]["log_cost"], places=12)
        self.assertAlmostEqual(G["B"]["C"]["log_cost_repaired"], G["B"]["C"]["log_cost"], places=12)

    def test_dijkstra_matches_floyd_warshall(self):
        # Build two identical copies and run the two methods
        G1 = self.make_graph_with_triangle_violation()
        G2 = self.make_graph_with_triangle_violation()

        tvc1 = TriangleViolationChecker(G1)
        tvc1.calculate_repaired_costs_floyd_warshall()

        tvc2 = TriangleViolationChecker(G2)
        tvc2.calculate_repaired_costs_dijkstra()

        # Compare repaired costs edge by edge
        for u, v in G1.edges():
            c1 = G1[u][v]["log_cost_repaired"]
            c2 = G2[u][v]["log_cost_repaired"]
            self.assertAlmostEqual(c1, c2, places=12)

            # flags consistent
            self.assertEqual(G1[u][v]["violation"], G2[u][v]["violation"])

    def test_transform_back_updates_weight_and_stores_similarity(self):
        G = self.make_graph_with_triangle_violation()
        tvc = TriangleViolationChecker(G)
        tvc.calculate_repaired_costs_floyd_warshall()
        tvc.transform_back_to_similarity()

        # A->C should now have weight = 0.81, repaired_similarity = 0.81
        ac = G["A"]["C"]
        repaired_similarity = ac["repaired_similarity"]
        self.assertAlmostEqual(repaired_similarity, 0.9 * 0.9, places=12)
        self.assertAlmostEqual(ac["weight"], repaired_similarity, places=12)

        # A->B should remain 0.9
        ab = G["A"]["B"]
        self.assertAlmostEqual(ab["repaired_similarity"], ab["weight"], places=12)
        self.assertAlmostEqual(ab["weight"], 0.9, places=12)

    def test_consistent_graph_no_change(self):
        G = self.make_graph_consistent_triangle()
        tvc = TriangleViolationChecker(G)
        tvc.calculate_repaired_costs_dijkstra()  # either method should keep things

        # No violations expected
        for _, _, d in G.edges(data=True):
            self.assertIn("log_cost_repaired", d)
            self.assertFalse(d["violation"])
            self.assertAlmostEqual(d["log_cost_repaired"], d["log_cost"], places=12)

        # Back transform preserves original weights
        original = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        tvc.transform_back_to_similarity()
        for u, v, d in G.edges(data=True):
            self.assertAlmostEqual(d["weight"], original[(u, v)], places=12)
            self.assertAlmostEqual(d["repaired_similarity"], original[(u, v)], places=12)

    def test_unreachable_nodes_do_not_fake_repairs(self):
        # Two components: A->B and C->D. No path A->D etc.
        G = SemanticMatchGraph()
        G.add_semantic_match("A", "B", 0.6)
        G.add_semantic_match("C", "D", 0.7)

        tvc = TriangleViolationChecker(G)
        tvc.calculate_repaired_costs_dijkstra()

        # Each edge should keep its direct cost; no violations, since no path can improve it
        for _, _, d in G.edges(data=True):
            self.assertIn("log_cost_repaired", d)
            self.assertAlmostEqual(d["log_cost_repaired"], d["log_cost"], places=12)
            self.assertFalse(d["violation"])


if __name__ == "__main__":
    unittest.main()
