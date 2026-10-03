"""
Integration tests for several Semantic Match Registries (SMRs) working together

The tests run against the compose stack in `compose.yaml`: Three SMRs (`smr-a`, `smr-b`, `smr-c`) and one SMR
discovery service. Each SMR only knows its own matching graph (`resources/smr-*/graph.json`) and the discovery
service. The discovery service maps the namespaces `a.example`, `b.example` and `c.example`, as well as the IRDI
prefixes `0112` and `0173`, to the SMRs (`resources/discovery/endpoints.json`).

Each test uses its own part of the graphs (e.g. `https://a.example/transitive/...`), so that the tests are
independent of each other.
"""
import os
import unittest

from tests import _stack


def setUpModule() -> None:
    _stack.up()


def tearDownModule() -> None:
    # Set `SMR_INTEGRATION_TESTS_KEEP_STACK` to inspect the services (e.g. their logs) after the tests
    if not os.environ.get("SMR_INTEGRATION_TESTS_KEEP_STACK"):
        _stack.down()


class TestFederatedQuery(unittest.TestCase):
    def test_transitive_match_across_registries_iri(self) -> None:
        """
        `smr-a` knows A -> B (0.8) and `smr-b` knows B -> C (0.5). A query for A at `smr-a` returns A -> B -> C with
        the product of the scores (0.4).
        """
        response = _stack.query_matches(_stack.SMR_A, "https://a.example/transitive/A", score_limit=0.1)
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            {
                "https://a.example/transitive/A -> https://b.example/transitive/B": 0.8,
                "https://a.example/transitive/A -> https://b.example/transitive/B -> https://b.example/transitive/C":
                    0.4,
            },
            _stack.paths(response.json())
        )
        for match in response.json():
            self.assertEqual("https://a.example/transitive/A", match["base_semantic_id"])

    def test_transitive_match_across_registries_irdi(self) -> None:
        """
        The same as `test_transitive_match_across_registries_iri`, but with IRDIs: `smr-a` knows A -> B (0.8) and
        `smr-b` knows B -> C (0.5). A query for A at `smr-a` returns A -> B -> C with the score 0.4.
        """
        response = _stack.query_matches(_stack.SMR_A, "0112-1ABC#01-TRA001#1", score_limit=0.1)
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            {
                "0112-1ABC#01-TRA001#1 -> 0173-1ABC#01-TRB001#1": 0.8,
                "0112-1ABC#01-TRA001#1 -> 0173-1ABC#01-TRB001#1 -> 0173-1ABC#01-TRC001#1": 0.4,
            },
            _stack.paths(response.json())
        )

    def test_score_limit_stops_at_remote_hop(self) -> None:
        """
        With a `score_limit` of 0.45, A -> B (0.8) is returned, but A -> B -> C (0.4) is not, even though B -> C
        (0.5) alone is above the limit at `smr-b`.
        """
        response = _stack.query_matches(_stack.SMR_A, "https://a.example/transitive/A", score_limit=0.45)
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            {"https://a.example/transitive/A -> https://b.example/transitive/B": 0.8},
            _stack.paths(response.json())
        )

    def test_local_only_queries_no_remote_registry(self) -> None:
        """
        With `local_only`, `smr-a` only returns its own match A -> B and does not ask `smr-b` for B -> C.
        """
        response = _stack.query_matches(
            _stack.SMR_A, "https://a.example/transitive/A", score_limit=0.1, local_only=True
        )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            {"https://a.example/transitive/A -> https://b.example/transitive/B": 0.8},
            _stack.paths(response.json())
        )

    def test_repeated_query_is_deterministic(self) -> None:
        """
        The same query across registries returns exactly the same result, including the order of the matches.
        """
        responses = [
            _stack.query_matches(_stack.SMR_A, "https://a.example/fault/A", score_limit=0.1) for _ in range(5)
        ]
        for response in responses:
            self.assertEqual(200, response.status_code, response.text)
            self.assertEqual(responses[0].json(), response.json())


class TestLoopAvoidance(unittest.TestCase):
    def test_cycle_across_registries_terminates(self) -> None:
        """
        `smr-a` knows X -> Y and `smr-b` knows Y -> X. A query for X at `smr-a` terminates, does not return
        duplicates and does not return the path back to X.
        """
        response = _stack.query_matches(_stack.SMR_A, "https://a.example/cycle/X", score_limit=0.1)
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            {"https://a.example/cycle/X -> https://b.example/cycle/Y": 0.9},
            _stack.paths(response.json())
        )
        self.assertEqual(1, len(response.json()))

    def test_chain_back_to_checked_registry(self) -> None:
        """
        Known limitation: `smr-a` knows A -> B and C -> D, `smr-b` knows B -> C. A query for A at `smr-a` finds
        A -> B -> C, but not A -> B -> C -> D: `smr-b` does not ask `smr-a` for C, since `smr-a` is already
        checked.
        """
        response = _stack.query_matches(_stack.SMR_A, "https://a.example/gap/A", score_limit=0.1)
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            {
                "https://a.example/gap/A -> https://b.example/gap/B": 0.9,
                "https://a.example/gap/A -> https://b.example/gap/B -> https://a.example/gap/C": 0.81,
            },
            _stack.paths(response.json())
        )


class TestDiscovery(unittest.TestCase):
    def test_remote_registry_found_via_discovery(self) -> None:
        """
        `smr-a` has no configuration about `smr-c`, it only knows the discovery service. The discovery service maps
        `c.example` to `smr-c`, so a query for A at `smr-a` returns A -> B (`smr-a`) -> C (`smr-c`).
        """
        discovery_response = _stack.query_smr("https://c.example/discovery/B")
        self.assertEqual(200, discovery_response.status_code, discovery_response.text)
        self.assertEqual("http://smr-c:8000", discovery_response.json()["smr_endpoint"])

        response = _stack.query_matches(_stack.SMR_A, "https://a.example/discovery/A", score_limit=0.1)
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            {
                "https://a.example/discovery/A -> https://c.example/discovery/B": 0.9,
                "https://a.example/discovery/A -> https://c.example/discovery/B -> https://c.example/discovery/C":
                    0.81,
            },
            _stack.paths(response.json())
        )


class TestFaultTolerance(unittest.TestCase):
    def test_unreachable_registry_is_skipped(self) -> None:
        """
        `smr-a` knows A -> B and A -> X, `smr-b` knows B -> C and `smr-c` knows X -> Y. With `smr-c` stopped, a
        query for A at `smr-a` still succeeds and returns all matches of `smr-a` and `smr-b`. Only X -> Y is missing.
        """
        _stack.stop("smr-c")
        try:
            response = _stack.query_matches(_stack.SMR_A, "https://a.example/fault/A", score_limit=0.1)
        finally:
            _stack.start("smr-c")
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            {
                "https://a.example/fault/A -> https://b.example/fault/B": 0.9,
                "https://a.example/fault/A -> https://c.example/fault/X": 0.9,
                "https://a.example/fault/A -> https://b.example/fault/B -> https://b.example/fault/C": 0.81,
            },
            _stack.paths(response.json())
        )


if __name__ == '__main__':
    unittest.main()
