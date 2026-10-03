"""
Helpers to control the compose stack defined in `compose.yaml` and to talk to its services
"""
import os
import subprocess
from typing import Any, Dict, List

import requests

COMPOSE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Host ports, as published in `compose.yaml`
SMR_A: str = "http://localhost:8001"
SMR_B: str = "http://localhost:8002"
SMR_C: str = "http://localhost:8003"
DISCOVERY: str = "http://localhost:8125"

# Timeout in seconds for each request of the tests
REQUEST_TIMEOUT: float = 30.0


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], cwd=COMPOSE_DIR, check=True)


def up() -> None:
    compose("up", "--detach", "--build", "--wait")


def down() -> None:
    compose("down", "--volumes")


def stop(service: str) -> None:
    compose("stop", service)


def start(service: str) -> None:
    compose("up", "--detach", "--wait", service)


def query_matches(
        smr: str,
        semantic_id: str,
        score_limit: float,
        local_only: bool = False
) -> requests.Response:
    return requests.post(
        f"{smr}/query_matches",
        json={"semantic_id": semantic_id, "score_limit": score_limit, "local_only": local_only},
        timeout=REQUEST_TIMEOUT,
    )


def query_smr(semantic_id: str) -> requests.Response:
    return requests.post(f"{DISCOVERY}/query_smr", json={"semantic_id": semantic_id}, timeout=REQUEST_TIMEOUT)


def paths(matches: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Reduce matches to `{"A -> B -> C": score}`, so that tests can compare whole paths at once
    """
    return {" -> ".join(m["path"] + [m["match_semantic_id"]]): round(m["score"], 10) for m in matches}
