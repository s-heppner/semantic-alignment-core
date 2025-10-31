import os
import multiprocessing

import requests
import unittest

from fastapi import FastAPI
import uvicorn

from smr import algorithm
from smr.service import SemanticMatchRegistry

from contextlib import contextmanager
import signal
import time

import json as js


def run_server():
    # Read in graph (new {"nodes":..., "edges":...} schema)
    match_graph = algorithm.SemanticMatchGraph.from_file(
        filename=os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "test_resources/example_graph.json"
        ))
    )

    # Init service
    semantic_matching_service = SemanticMatchRegistry(
        endpoint="localhost",
        graph=match_graph
    )

    # Mock resolver
    def mock_get_matcher(self, semantic_id):
        return "http://remote-service:8000"
    SemanticMatchRegistry._get_matcher_from_semantic_id = mock_get_matcher

    # Mock remote service POST /query_matches to return a list of matches
    original_requests_post = requests.post

    class SimpleResponse:
        def __init__(self, obj, status_code=200):
            self._obj = obj
            self.status_code = status_code

        def json(self):
            return self._obj

    def mock_requests_post(url, json=None, **kwargs):
        if url == "http://remote-service:8000/query_matches":
            match_one = algorithm.SemanticMatch(
                base_semantic_id="s-heppner.com/semanticID/three",
                match_semantic_id="remote-service.com/semanticID/tres",
                score=1.0,
                path=[]
            )
            # Server expects a plain list response now
            return SimpleResponse([match_one.model_dump()])
        return original_requests_post(url, json=json, **kwargs)

    requests.post = mock_requests_post

    # Run server
    app = FastAPI()
    app.include_router(semantic_matching_service.router)
    uvicorn.run(app, host="localhost", port=8000, log_level="error")


@contextmanager
def run_server_context():
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    try:
        time.sleep(2)  # Wait for the server to start
        yield
    finally:
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            os.kill(server_process.pid, signal.SIGKILL)
            server_process.join()


class TestSemanticMatchRegistry(unittest.TestCase):
    def test_post_matches(self):
        with run_server_context():
            new_match = {
                "base_semantic_id": "https://s-heppner.com/semantic_id/new",
                "match_semantic_id": "https://s-heppner.com/semantic_id/nouveaux",
                "score": 0.95,
                "path": [],
            }
            response = requests.post(
                "http://localhost:8000/post_matches",
                json=[new_match]
            )
            self.assertEqual(200, response.status_code)
            # Todo: Make sure this does not become a problem in other tests

    def test_get_all_matches(self):
        with run_server_context():
            resp = requests.get("http://localhost:8000/all_matches")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            # It now returns SemanticMatch dicts including metric_id and graph_score
            self.assertIsInstance(data, list)
            self.assertTrue(all(isinstance(x, dict) for x in data))

            # We expect at least these direct edges to exist (ignoring metric_id multiplicity):
            must_have = {
                ("https://s-heppner.com/semantic_id/one", "https://s-heppner.com/semantic_id/uno", 0.9),
                ("https://s-heppner.com/semantic_id/one", "https://remote.com/semantic_id/deux", 0.7),
                ("https://s-heppner.com/semantic_id/uno", "https://s-heppner.com/semantic_id/trois", 0.6),
            }
            triples = {(d["base_semantic_id"], d["match_semantic_id"], round(float(d["score"]), 10)) for d in data}
            for t in must_have:
                self.assertIn(t, triples, f"Missing expected match {t}")

            # graph_score should always be present and numeric
            for d in data:
                self.assertIn("graph_score", d)
                self.assertIsInstance(d["graph_score"], (int, float))

    def test_get_matches_local_and_remote(self):
        with run_server_context():
            match_request = {
                "semantic_id": "https://s-heppner.com/semantic_id/one",
                "score_limit": 0.7,
                "local_only": False
            }
            resp = requests.post("http://localhost:8000/query_matches", json=match_request)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()

            # Must include the two local single-edge matches >= 0.7
            want_local = {
                ("https://s-heppner.com/semantic_id/one", "https://s-heppner.com/semantic_id/uno", 0.9),
                ("https://s-heppner.com/semantic_id/one", "https://remote.com/semantic_id/deux", 0.7),
            }
            triples = {(d["base_semantic_id"], d["match_semantic_id"], round(float(d["score"]), 10)) for d in data}
            for t in want_local:
                self.assertIn(t, triples)

            # Remote mock may add extra entries; just ensure list shape is sane
            self.assertTrue(all(isinstance(x, dict) for x in data))

    def test_get_matches_no_matches(self):
        with run_server_context():
            match_request = {
                "semantic_id": "s-heppner.com/semanticID/unknown",
                "score_limit": 0.5,
                "local_only": True
            }
            resp = requests.post("http://localhost:8000/query_matches", json=match_request)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual([], resp.json())


if __name__ == '__main__':
    unittest.main()
