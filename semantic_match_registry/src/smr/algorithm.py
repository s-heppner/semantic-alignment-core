import json
from typing import List, Tuple, Optional
import heapq

import networkx as nx
from pydantic import BaseModel


class SemanticMatchGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()

    def add_semantic_match(self,
                           base_semantic_id: str,
                           match_semantic_id: str,
                           score: float,
                           metric_id: Optional[str] = None) -> None:
        """
        Add a semantic match to the graph.

        This will overwrite the match with the same `metric_id` (including `None`), if it exists.
        Every `{metric_id: score}` will be stored in the `metric_scores` edge attribute, the actual `weight` of the
        edge will be the maximum of the scores, as that leads to minimal
        semantic similarity triangle inequality violations.

        :param base_semantic_id: From semantic_id
        :param match_semantic_id: To semantic_id (Note, the graph is not necessarily symmetric!)
        :param score: Semantic similarity score
        :param metric_id: Globally identifying string of the metric used (e.g. the NLP model + version)
        """
        u, v = base_semantic_id, match_semantic_id
        # Create edge if it doesn't exist yet
        if not self.has_edge(u, v):
            self.add_edge(u, v, weight=float(score), metric_scores={})
        # Retrieve metrics dict
        metrics = self[u][v].setdefault("metric_scores", {})
        # Use a fallback key if no metric_id was provided
        key = metric_id if metric_id is not None else "_no_metric_id"
        metrics[key] = float(score)
        # Update the working weight to the max of all metrics
        self[u][v]["weight"] = max(metrics.values())

    def get_all_matches(self) -> List["SemanticMatch"]:
        matches: List["SemanticMatch"] = []

        # Iterate over all edges in the graph
        for base, match, data in self.edges(data=True):
            weight = data.get("weight", 0.0)  # Get weight, default to 0.0 if missing
            for metric, score in data.get("metric_scores", {}).items():
                matches.append(SemanticMatch(
                    base_semantic_id=base,
                    match_semantic_id=match,
                    score=score,
                    path=[],  # Direct match, no intermediate nodes
                    metric_id=None if metric == "_no_metric_id" else metric,
                    graph_score=weight,
                ))

        return matches

    def to_file(self, filename: str) -> None:
        data = {
            "nodes": list(self.nodes()),
            "edges": [
                {
                    "u": u,
                    "v": v,
                    "weight": float(d.get("weight", 0.0)),
                    "metric_scores": {k: float(vv) for k, vv in d.get("metric_scores", {}).items()},
                }
                for u, v, d in self.edges(data=True)
            ],
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, filename: str) -> "SemanticMatchGraph":
        with open(filename) as f:
            data = json.load(f)
        G = cls()
        G.add_nodes_from(data.get("nodes", []))
        for e in data.get("edges", []):
            u, v = e["u"], e["v"]
            metrics = e.get("metric_scores", {})
            # rebuild edge and recalc active weight as max(metric_scores)
            G.add_edge(u, v, metric_scores=metrics,
                       weight=max(metrics.values()) if metrics else float(e.get("weight", 0.0)))
        return G


class SemanticMatch(BaseModel):
    base_semantic_id: str
    match_semantic_id: str
    score: float
    path: List[str]  # The path of `semantic_id`s that the algorithm took
    metric_id: Optional[str] = None  # Globally identifying string of the metric used (e.g. the NLP model + version)
    graph_score: Optional[float] = None  # The score that was considered in the SemanticMatchGraph's search algorithm
    # This value can be ignored in most cases and is more interesting as a debug information. It might explain why a
    # SemanticMatch with this score was returned by the query, even though the score might not fit.

    def __str__(self) -> str:
        return f"{' -> '.join(self.path + [self.match_semantic_id])} = {self.score}"

    def __hash__(self):
        return hash((
            self.base_semantic_id,
            self.match_semantic_id,
            self.score,
            tuple(self.path),
            self.metric_id,
            self.graph_score,
        ))


def find_semantic_matches(
    graph: SemanticMatchGraph,
    semantic_id: str,
    min_score: float = 0.5
) -> List[SemanticMatch]:
    """
    Find semantic matches for a given node with a minimum score threshold.

    Args:
        graph (nx.DiGraph): The directed graph with weighted edges.
        semantic_id (str): The starting semantic_id.
        min_score (float): The minimum similarity score to consider.
            This value is necessary to ensure the search terminates also with sufficiently large graphs.

    Returns:
        List[SemanticMatch]:
        A list of MatchResults, sorted by their score with the highest score first.
    """
    if semantic_id not in graph:
        return []

    # We need to make sure that all possible paths starting from the given semantic_id are explored.
    # To achieve this, we use the concept of "priority queue". While we could use a simple FIFO list of matches to
    # explore, this way we actually end up with an already sorted result with the highest match at the beginning of the
    # list. As possible implementation of this abstract data structure, we choose to use a "max-heap".
    # However, there is no efficient implementation of a max-heap in Python, so rather we use the built-in "min-heap"
    # and negate the score values. A priority queue ensures that elements with the highest priority are processed first,
    # regardless of when they were added.
    # We initialize the priority queue:
    pq: List[Tuple[float, str, List[str]]] = [(-1.0, semantic_id, [])]  # (neg_score, node, path)
    # The queue is structured as follows:
    #   - `neg_score`: The negative score of the match
    #   - `node`: The `match_semantic_id` of the match
    #   - `path`: The path between the `semantic_id` and the `match_semantic_id`

    # Prepare the result list
    results: List[SemanticMatch] = []

    # Run the priority queue until all possible paths have been explored
    # This means in each iteration:
    #   - We pop the top element of the queue as it's the next highest semantic match we want to explore
    #   - If the match has a score higher or equal to the given `min_score`, we add it to the results
    #   - We add all connected `semantic_id`s to the priority queue to be treated next
    #   - We go to the next element of the queue
    while pq:
        # Get the highest-score match from the queue
        neg_score, node, path = heapq.heappop(pq)
        score = -neg_score  # Convert back to positive

        # Store result if above threshold (except the start node)
        if node != semantic_id and score >= min_score:
            # Single-edge path if path == [semantic_id]
            is_single_edge = (len(path) == 1 and path[0] == semantic_id)

            if is_single_edge:
                # Emit one result per metric on the direct edge (semantic_id -> node)
                edge_data = graph[semantic_id].get(node, {})
                metrics = edge_data.get("metric_scores", {})
                if metrics:
                    for metric, mscore in metrics.items():
                        matches_metric_id = None if metric == "_no_metric_id" else metric
                        # score = metric's own score; graph_score = path product (ranking basis)
                        results.append(SemanticMatch(
                            base_semantic_id=semantic_id,
                            match_semantic_id=node,
                            score=float(mscore),
                            path=path,  # single hop path holds [semantic_id]
                            metric_id=matches_metric_id,
                            graph_score=score,
                        ))
                else:
                    # No metrics dict present, fall back to aggregated-only result
                    results.append(SemanticMatch(
                        base_semantic_id=semantic_id,
                        match_semantic_id=node,
                        score=score,
                        path=path,
                        metric_id=None,
                        graph_score=score,
                    ))
            else:
                # Multi-edge path: no single metric applies
                results.append(SemanticMatch(
                    base_semantic_id=semantic_id,
                    match_semantic_id=node,
                    score=score,  # path product
                    path=path,
                    metric_id=None,
                    graph_score=score,
                ))

        # Traverse to the neighboring and therefore connected `semantic_id`s
        for neighbor, edge_data in graph[node].items():
            # Avoid any cycle: no revisiting nodes that are already in this path
            # Note: `path` holds all previous nodes; `node` is *not* yet in `path`
            # This also lets us avoid immediate backtrack A->B->A ping-pong for free
            if neighbor in path or neighbor == node:
                continue

            edge_weight = float(edge_data.get("weight", 0.0))
            new_score: float = score * edge_weight  # Multiplicative propagation

            # Prevent loops by ensuring we do not revisit the start node after the first iteration
            if neighbor == semantic_id:
                continue  # Avoid re-exploring the start node

            # We add the newly found `semantic_id`s to the queue to be explored next in order of their score
            if new_score >= min_score:
                heapq.heappush(pq, (-new_score, neighbor, path + [node]))  # Push updated path

    return results
