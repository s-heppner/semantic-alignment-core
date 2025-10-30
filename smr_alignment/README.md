# SMR Alignment

This package contains tools for checking the alignment of data to the requirements of the Semantic Match Registry 
concept.

## Check for Semantic Similarity Triangle Violations

A tool for finding *Semantic Similarity Triangle Violations* is located in: `triangle_violation_checker.py`.

You can use it as follows, to check a given `smr.algorithm.SemanticMatchGraph` for violations:

```python
from smr.algorithm import SemanticMatchGraph
from smr_alignment.triangle_violation_checker import TriangleViolationChecker

smg: SemanticMatchGraph = SemanticMatchGraph()  # Your SemanticMatchGraph
tvc: TriangleViolationChecker = TriangleViolationChecker(smg)

# The TriangleViolationChecker will automatically calculate the log-costs of the SemanticMatchGraph 
# during initialization.
# You then have to decide whether to use Floyd-Warshall or Dijkstra for calculating the repaired costs 
# (and therefore the violations):

tvc.calculate_repaired_costs_floyd_warshall()  # Use with dense (and small-ish) graphs
tvc.calculate_repaired_costs_dijkstra()  # Use with sparse (and larger) graphs

# Both algorithms are tested to give the same results (up to 12 places behind the comma)

# Finally: Transform back to similarity space for the repaired similarities:
tvc.transform_back_to_similarity()

# Now, smg has the repaired weights:

for u,v, data in smg.edges(data=True):
    print(data["weight"])
    
    # Furthermore, we have some more practical data for each edge:
    print(data["original_similarity"])  # The original, unmodified similarity
    print(data["log_cost"])             # The similarity score transformed into log space
    print(data["log_cost_repaired"])    # The repaired log_cost, made fit with the inherent semantic model of the graph
    print(data["violation"])            # True, if a violation was found else False
    print(data["repaired_similarity"])  # The repaired similarity cost (should be equal to "weight")
```
