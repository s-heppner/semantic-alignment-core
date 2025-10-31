"""
This module implements a checker to flag all semantic similarity triangle violations, e.g. cases where:

s(A, C) <= s(A, B) * s(B, C)

These values violate the inherent semantic model of the graph and lead to it being not max-product transitive,
a requirement for the SMR concept.

For more details, I'd like to refer to my dissertation.
"""
import math

from tqdm import tqdm
import networkx as nx

from smr.algorithm import SemanticMatchGraph


class TriangleViolationChecker:
    def __init__(self, semantic_match_graph: SemanticMatchGraph) -> None:
        self.semantic_match_graph: SemanticMatchGraph = semantic_match_graph
        self.add_log_cost_to_graph()

    def add_log_cost_to_graph(self, epsilon: float = 1e-8) -> None:
        """
        Adds a log-space 'log_cost' edge attribute to the graph, in place.

        Clamps edge weights between (epsilon, 1 - epsilon) to avoid
        logarithmic singularities.
        """
        edges = tqdm(
            self.semantic_match_graph.edges(data=True),
            total=self.semantic_match_graph.number_of_edges(),
            desc="Transforming to log-space",
        )

        for u, v, data in edges:
            # get original similarity
            s = float(data["weight"])
            data["original_similarity"] = s

            # clamp into (epsilon, 1 - epsilon)
            s_clamped = min(max(s, epsilon), 1 - epsilon)

            # avoid recomputing max() inside log, since it's already clamped
            data["log_cost"] = -math.log(s_clamped)

    def calculate_repaired_costs_floyd_warshall(self) -> None:
        """
        Repair costs using all-pairs shortest paths (Floyd–Warshall).
        Sets log_cost_repaired = min(direct_cost, shortest_path_cost) in the graph.

        Note: Floyd–Warshall is simple and cubic in node count. Use it when the graph is small-ish or fairly dense.
        """
        print("Calculating distances using Floyd-Warshall. This may take a while (O(n^3)).")
        distances = dict(nx.floyd_warshall(self.semantic_match_graph, weight="log_cost"))

        # Apply repairs with a progress bar over edges
        edges = tqdm(
            self.semantic_match_graph.edges(data=True),
            total=self.semantic_match_graph.number_of_edges(),
            desc="Applying Floyd-Warshall repaired costs",
        )
        for u, v, data in edges:
            direct = float(data["log_cost"])
            shortest = distances[u][v]  # float('inf') if disconnected
            repaired = min(direct, shortest)
            data["log_cost_repaired"] = repaired
            data["violation"] = True if repaired < direct else False

    def calculate_repaired_costs_dijkstra(self) -> None:
        """
        Repair costs using repeated single-source Dijkstra.
        Sets log_cost_repaired = min(direct_cost, shortest_path_cost) in the graph.

        Note: Better for large, sparse graphs. Roughly O(m * log n) per source.
        """
        nodes = self.semantic_match_graph.nodes

        for u in tqdm(nodes, total=len(nodes), desc="Running Dijkstra per node"):
            dists = nx.single_source_dijkstra_path_length(self.semantic_match_graph, source=u, weight="log_cost")
            for _, v, data in self.semantic_match_graph.out_edges(u, data=True):
                direct = float(data["log_cost"])
                shortest = dists.get(v, float("inf"))  # inf if unreachable
                repaired = min(direct, shortest)
                data["log_cost_repaired"] = repaired
                data["violation"] = True if repaired < direct else False

    def transform_back_to_similarity(self) -> None:
        """
        Transforms repaired log-costs back into similarity scores.

        For each edge:
            repaired_similarity = exp(-log_cost_repaired)
            weight = repaired_similarity
        """
        edges = tqdm(
            self.semantic_match_graph.edges(data=True),
            total=self.semantic_match_graph.number_of_edges(),
            desc="Transforming back to similarity space",
        )

        for _, _, data in edges:
            # Retrieve the repaired log-cost (if absent, skip)
            if "log_cost_repaired" not in data:
                continue

            repaired_cost = float(data["log_cost_repaired"])
            repaired_similarity = math.exp(-repaired_cost)

            # Store for reference and update active weight
            data["repaired_similarity"] = repaired_similarity
            data["weight"] = repaired_similarity
