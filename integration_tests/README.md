# Integration Tests

These tests run several Semantic Match Registries (SMRs) and the SMR discovery service together and query them over
HTTP. They check that matches are found across registries, that the discovery service routes the queries, and that
an unreachable registry does not break a query.

## Structure
- `compose.yaml` starts three SMRs (`smr-a`, `smr-b`, `smr-c`) and one SMR discovery service (`smr-discovery`),
  built from [semantic_match_registry](../semantic_match_registry) and [smr_discovery](../smr_discovery).
- `resources/smr-*/` holds the `config.ini` and the matching graph (`graph.json`) of each SMR.
  Each SMR only knows its own graph and the discovery service.
- `resources/discovery/endpoints.json` maps the namespaces `a.example`, `b.example`, `c.example` and the IRDI
  prefixes `0112` and `0173` to the SMRs. No lookup goes to DNS, so the tests do not depend on the network.
- `tests/test_federation.py` holds the test cases. The docstring of each test describes the scenario.

## Test Overview
All tests send their query to `smr-a`. `smr-a` does not know the other SMRs. For each match it finds, it asks the
discovery service which SMR is responsible for the matched ID, and then asks that SMR for further matches. The
other SMRs do the same in turn.

```
          query
            │
            ▼
        ┌───────┐   1. Which SMR is responsible?   ┌───────────────┐
        │ smr-a │ ───────────────────────────────► │ smr-discovery │
        └───────┘                                  └───────────────┘
            │
            │ 2. Any further matches?
            ▼
   ┌────────┴────────┐
┌───────┐        ┌───────┐
│ smr-b │        │ smr-c │
└───────┘        └───────┘
```

In the tables below, `A -> B (0.8)` means that the SMR in the column stores a match from `A` to `B` with the score 0.8.
The score of a path is the product of the scores along the path, so `A -> B -> C` has the score 0.8 · 0.5 = 0.4.
Each test uses its own IDs (e.g. `https://a.example/transitive/A`), so the tests do not interfere with each other.

### Matching across registries (`TestFederatedQuery`)

| Test | `smr-a` | `smr-b` | Query | Expected result |
|---|---|---|---|---|
| `test_transitive_match_across_registries_iri` | `A -> B (0.8)` | `B -> C (0.5)` | `A`, limit 0.1 | `A -> B (0.8)`, `A -> B -> C (0.4)` |
| `test_transitive_match_across_registries_irdi` | `A -> B (0.8)` | `B -> C (0.5)` | `A`, limit 0.1 | The same, with IRDIs instead of IRIs |
| `test_score_limit_stops_at_remote_hop` | `A -> B (0.8)` | `B -> C (0.5)` | `A`, limit 0.45 | Only `A -> B (0.8)`, since 0.4 < 0.45 |
| `test_local_only_queries_no_remote_registry` | `A -> B (0.8)` | `B -> C (0.5)` | `A`, `local_only` | Only `A -> B (0.8)`, `smr-b` is not asked |
| `test_repeated_query_is_deterministic` | (as in the fault test) | | `A`, 5 times | 5 identical results, in the same order |

### Loops between registries (`TestLoopAvoidance`)

| Test | `smr-a` | `smr-b` | Query | Expected result |
|---|---|---|---|---|
| `test_cycle_across_registries_terminates` | `X -> Y (0.9)` | `Y -> X (0.9)` | `X` | The query terminates and returns only `X -> Y`, without duplicates and without the path back to `X` |
| `test_chain_back_to_checked_registry` | `A -> B`, `C -> D` (each 0.9) | `B -> C (0.9)` | `A` | `A -> B` and `A -> B -> C`, but **not** `A -> B -> C -> D` |

The second test documents a known limitation: `smr-b` does not ask `smr-a` for `C`, because `smr-a` is already in
the list of checked registries. This is what keeps the query from going in circles, but it also means that a chain
that comes back to an already checked registry is not followed further.

### Routing via the discovery service (`TestDiscovery`)

| Test | `smr-a` | `smr-c` | Query | Expected result |
|---|---|---|---|---|
| `test_remote_registry_found_via_discovery` | `A -> B (0.9)` | `B -> C (0.9)` | `A` | `A -> B (0.9)`, `A -> B -> C (0.81)` |

`smr-a` is not configured with any information about `smr-c`. The test first checks that the discovery service
returns `smr-c` for `B`, and then that `smr-a` finds `C` through it.

### Unreachable registry (`TestFaultTolerance`)

| Test | `smr-a` | `smr-b` | `smr-c` | Query | Expected result |
|---|---|---|---|---|---|
| `test_unreachable_registry_is_skipped` | `A -> B`, `A -> X` | `B -> C` | `X -> Y` (stopped) | `A` | `A -> B`, `A -> X`, `A -> B -> C`, and no error |

All scores are 0.9. The test stops `smr-c` before the query and starts it again afterwards. The query still
succeeds and returns everything that `smr-a` and `smr-b` know. Only `A -> X -> Y` is missing.

## How to Use
You need a working Python installation (3.11 or higher) and Docker with the compose plugin.
The tests start the compose stack themselves (`docker compose up --build --wait`) and remove it again at the end.
They use the host ports 8001, 8002, 8003 and 8125.

```commandline
cd integration_tests
python3 -m venv venv
source venv/bin/activate
pip install .[dev]
python -m unittest discover -v
```

To look at the services by hand, start the stack with `docker compose up --build --wait` and query, for example,
`smr-a`:

```commandline
curl -X POST http://localhost:8001/query_matches -H 'Content-Type: application/json' \
  -d '{"semantic_id": "https://a.example/transitive/A", "score_limit": 0.1, "local_only": false}'
```

Stop it again with `docker compose down`.

If `SMR_INTEGRATION_TESTS_KEEP_STACK` is set, the tests leave the stack running afterwards, for example to
read the logs with `docker compose logs`.
