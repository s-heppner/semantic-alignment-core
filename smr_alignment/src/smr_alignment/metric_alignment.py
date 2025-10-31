"""
Optimize a given graph with multiple metrics to minimize triangle violations.

How to use:

# 1) Your graph already has per-edge metric_scores
mao = MetricAlignmentOptimizer(G)

# 2) Learn weights (sub-sample triplets if the graph is big)
w = mao.optimize_weights(eps=1e-8, sample_triplets=50_000, entropy_lambda=1e-3)

# 3) Apply unified scores (and make them the working weight)
mao.apply_unified_scores(weights=w, replace_weight=True)

"""
import math
from typing import Dict, List, Optional, Sequence, Tuple
from itertools import combinations

import numpy as np
import cvxpy as cp
from tqdm import tqdm
import networkx as nx


class MetricAlignmentOptimizer:
    """
    Learn metric weights w (w>=0, sum w=1) that minimize the sum of
    triangle-violation margins in log-space, then apply unified scores.

    Assumes each edge (u,v) has:
        - d["metric_scores"] : Dict[metric_id, score in [0,1]]
        - d["weight"]        : working score (will optionally be overwritten)
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.metrics_: Optional[List[str]] = None
        self.weights_: Optional[Dict[str, float]] = None

    # ---------- utilities ----------

    @staticmethod
    def _clamp_score(x: float, epsilon: float) -> float:
        """
        Clamp a given (score) value between (epsilon, 1-epsilon)
        """
        return min(max(float(x), epsilon), 1.0 - epsilon)

    def _collect_pair_logs(
        self, eps: float = 1e-8, attr: str = "log_scores_by_metric"
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Build (u,v) -> {metric: -log(clamped score)} and cache on edge as attr.
        """
        pair_logs: Dict[Tuple[str, str], Dict[str, float]] = {}
        for u, v, d in tqdm(self.G.edges(data=True), desc="Collecting log scores"):
            ms = d.get("metric_scores", {})
            logs = {m: -math.log(self._clamp_score(s, eps)) for m, s in ms.items()}
            d[attr] = logs
            pair_logs[(u, v)] = logs
        # global metric set
        self.metrics_ = sorted({m for logs in pair_logs.values() for m in logs})
        return pair_logs

    def _alpha_vectors(
        self,
        pair_logs: Dict[Tuple[str, str], Dict[str, float]],
        metrics: Sequence[str],
        sample_triplets: Optional[int] = None,
        both_orientations: bool = True,
    ) -> np.ndarray:
        """
        Build alpha vectors for triplets, representing total violation of a metric. For a directed triple (A,B,C):
            alpha_m = c_m(A,C) - c_m(A,B) - c_m(B,C)
        If a metric is missing on any of those pairs, contribute 0 for that metric.
        Only constructs alphas for triplets where all three *pairs* exist in the graph.
        """
        nodes = list(self.G.nodes())
        if len(nodes) < 3:
            return np.zeros((0, len(metrics)))

        # Only consider triplets where the three directed pairs exist
        def pairs_exist(A, B, C):
            return ((A, B) in pair_logs) and ((B, C) in pair_logs) and ((A, C) in pair_logs)

        triplets = [(A, B, C) for A, B, C in combinations(nodes, 3)]
        if sample_triplets and len(triplets) > sample_triplets:
            stride = max(1, len(triplets) // sample_triplets)
            triplets = triplets[::stride]

        alphas: List[List[float]] = []
        for A, B, C in tqdm(triplets, desc="Building alpha vectors"):
            # consider both A->B->C and C->B->A (directed graph)
            orientations = [(A, B, C), (C, B, A)] if both_orientations else [(A, B, C)]
            for X, Y, Z in orientations:
                if not pairs_exist(X, Y, Z):
                    continue
                c_xy = pair_logs[(X, Y)]
                c_yz = pair_logs[(Y, Z)]
                c_xz = pair_logs[(X, Z)]
                alpha = []
                for m in metrics:
                    cxz = c_xz.get(m)
                    cxy = c_xy.get(m)
                    cyz = c_yz.get(m)
                    if cxz is None or cxy is None or cyz is None:
                        alpha.append(0.0)  # ignore missing metric for this triple
                    else:
                        alpha.append(cxz - (cxy + cyz))
                # skip all-zero vector? not necessary; harmless to keep
                alphas.append(alpha)
        if not alphas:
            return np.zeros((0, len(metrics)))
        return np.asarray(alphas, dtype=float)

    # ---------- optimization ----------

    def optimize_weights(
        self,
        eps: float = 1e-8,
        sample_triplets: Optional[int] = None,
        entropy_lambda: float = 0.0,
        solver: str = "ECOS",
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Solve: minimize sum_i pos(alpha_i · w) + entropy_lambda * sum w_i log w_i
        s.t. w >= 0, sum w = 1
        """
        pair_logs = self._collect_pair_logs(eps)
        metrics = self.metrics_ or []
        if not metrics:
            # No metric data at all; fall back to uniform singleton
            self.weights_ = {}
            return {}

        A = self._alpha_vectors(pair_logs, metrics, sample_triplets=sample_triplets)
        if A.shape[0] == 0:
            # no usable triplets; uniform weights
            w = np.ones(len(metrics)) / len(metrics)
            self.weights_ = {m: float(w[i]) for i, m in enumerate(metrics)}
            return self.weights_

        w = cp.Variable(len(metrics), nonneg=True)  # type: ignore
        margins = A @ w
        loss = cp.sum(cp.pos(margins))  # piecewise-linear convex

        if entropy_lambda > 0:
            # cp.entr(x) = x*log(x); we add small offset to keep it defined near zero
            loss += entropy_lambda * cp.sum(cp.entr(w + 1e-16))

        constraints = [cp.sum(w) == 1.0]  # cp.sum(w) == 1.0 in CVXPY doesn’t return a plain bool at runtime,
        # it returns a Constraint object, confusing type checkers. Therefore, the "type: ignore" below.
        prob = cp.Problem(cp.Minimize(loss), constraints)  # type: ignore

        try:
            prob.solve(solver=solver, verbose=verbose)
        except Exception:
            prob.solve(solver="SCS", verbose=verbose)

        if w.value is None:  # type: ignore
            raise RuntimeError("Weight optimization failed; check solver output/logs.")

        w_arr = np.clip(w.value, 0.0, None)  # type: ignore
        s = float(w_arr.sum())
        if s <= 0:
            w_arr = np.ones_like(w_arr) / len(w_arr)
        else:
            w_arr /= s

        self.weights_ = {m: float(w_arr[i]) for i, m in enumerate(metrics)}
        return self.weights_

    # ---------- apply unified score ----------

    def apply_unified_scores(
            self,
            weights: Optional[Dict[str, float]] = None,
            eps: float = 1e-8,
            out_cost_attr: str = "unified_log_cost",
            out_sim_attr: str = "unified_similarity",
            replace_weight: bool = True,
    ) -> None:
        """
        For each edge (u,v), compute:
          c_hat = sum_m w_m * (-log clamp(s_m))
          s_hat = exp(-c_hat) = prod_m clamp(s_m)^{w_m}

        Also store:
          - metric_scores_weighted: {m: clamp(s_m)^{w_m}}   (skip w_m == 0)
          - metric_weights:         {m: w_m}                (only metrics present on this edge)

        If replace_weight is True, set edge 'weight' to max(metric_scores_weighted.values())
        (i.e., the strongest weighted metric for that edge).
        """
        if weights is None:
            if self.weights_ is None:
                raise ValueError("No weights supplied and optimize_weights() has not been run.")
            weights = self.weights_

        for _, _, d in tqdm(self.G.edges(data=True), desc="Applying unified scores"):
            ms = d.get("metric_scores", {})
            if not ms:
                continue

            # Per-edge subset of weights (only for metrics present here)
            edge_w = {m: float(weights.get(m, 0.0)) for m in ms.keys() if weights.get(m, 0.0) > 0.0}
            d["metric_weights"] = edge_w  # report what we used on this edge

            # Unified cost/similarity (log-space blend = weighted geometric mean in real space)
            c_hat = 0.0
            for m, w_m in edge_w.items():
                s_m = self._clamp_score(ms[m], eps)  # or self._clamp01(...)
                c_hat += w_m * (-math.log(s_m))
            s_hat = math.exp(-c_hat)

            d[out_cost_attr] = c_hat
            d[out_sim_attr] = s_hat

            # Per-metric weighted scores in real space: s_m^{w_m}
            metric_scores_weighted = {}
            for m, w_m in edge_w.items():
                s_m = self._clamp_score(ms[m], eps)
                # Avoid s_m ** 0 = 1.0 inflation by skipping zero-weight metrics (already filtered)
                metric_scores_weighted[m] = s_m ** w_m
            d["metric_scores_weighted"] = metric_scores_weighted

            # Working weight: max of weighted metric scores (your requested policy)
            if replace_weight and metric_scores_weighted:
                d["weight"] = max(metric_scores_weighted.values())
