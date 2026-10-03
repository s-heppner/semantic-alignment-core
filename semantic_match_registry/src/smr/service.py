from typing import Optional, List, Set

from pydantic import BaseModel
import requests
from fastapi import APIRouter, Response

from smr import algorithm

# Timeout in seconds for requests to remote Semantic Match Registries and the discovery service
REMOTE_REQUEST_TIMEOUT: float = 5.0


class MatchRequest(BaseModel):
    """
    Request body for the :func:`service.SemanticMatchingService.get_match`

    :ivar semantic_id: The semantic ID that we want to find matches for
    :ivar score_limit: The minimum semantic similarity score to look for. Is considered as larger or equal (>=)
    :ivar local_only: If `True`, only check at the local service and do not request other services
    :ivar already_checked_locations: Optional Set of already checked Semantic Match Registries to avoid looping
    """
    semantic_id: str
    score_limit: float
    local_only: bool = True
    already_checked_locations: Optional[Set[str]] = None


class SemanticMatchRegistry:
    """
    A Semantic Match Registry

    It offers two operations:

    :func:`~.SemanticMatchRegistry.post_matches` allows to post
    :class:`model.SemanticMatch`es to the :class:`~.SemanticMatchRegistry`.

    :func:`~.SemanticMatchRegistry.query_matches` lets users get the
    :class:`model.SemanticMatch`es of the :class:`~.SemanticMatchRegistry`
    and the respective remote :class:`~.SemanticMatchRegistry`s.

    Additionally, the internal function
    :func:`~.SemanticMatchRegistry._get_matcher_from_semantic_id` lets the
    :class:`~.SemanticMatchRegistry` find the suiting remote
    :class:`~.SemanticMatchRegistry`s to a given `semantic_id`.
    """
    def __init__(
            self,
            endpoint: str,
            graph: algorithm.SemanticMatchGraph
    ):
        """
        Initializer of :class:`~.SemanticMatchRegistry`

        :ivar endpoint: The endpoint on which the service listens
        :ivar equivalences: The :class:`model.EquivalenceTable` of the semantic
            equivalences that this :class:`~.SemanticMatchRegistry` contains.
        """
        self.router = APIRouter()

        self.router.add_api_route(
            "/all_matches",
            self.get_all_matches,
            response_model=List[algorithm.SemanticMatch],
            methods=["GET"]
        )
        self.router.add_api_route(
            "/query_matches",
            self.query_matches,
            response_model=List[algorithm.SemanticMatch],
            methods=["POST"]
        )
        self.router.add_api_route(
            "/post_matches",
            self.post_matches,
            methods=["POST"]
        )
        self.endpoint: str = endpoint
        self.graph: algorithm.SemanticMatchGraph = graph

    def get_all_matches(self):
        """
        Returns all matches stored in the equivalence table-
        """
        matches = self.graph.get_all_matches()
        return matches

    def query_matches(
            self,
            request_body: MatchRequest
    ) -> List[algorithm.SemanticMatch]:
        """
        A query to find suiting :class:`algorithm.SemanticMatch`es for a given :class:`~.MatchRequest`.

        Returns a List of :class:`algorithm.SemanticMatch`es
        """
        # Try first local matching
        matches: List[algorithm.SemanticMatch] = algorithm.find_semantic_matches(
            graph=self.graph,
            semantic_id=request_body.semantic_id,
            min_score=request_body.score_limit
        )
        # If the request asks us to only locally look, we're done already
        if request_body.local_only:
            return matches
        # Now look for remote matches:
        additional_remote_matches: List[algorithm.SemanticMatch] = []
        for match in matches:
            # We need to make sure we do not go to the same Semantic Match Registry twice.
            # For that we update the already_checked_locations with the current endpoint:
            already_checked_locations: Set[str] = {self.endpoint}
            if request_body.already_checked_locations:
                already_checked_locations.update(request_body.already_checked_locations)

            # The discovery service decides which Semantic Match Registry is responsible for the `match_semantic_id`.
            # If that is this one, it is part of `already_checked_locations`, so we do not need to compare namespaces.
            remote_matching_service = self._get_matcher_from_semantic_id(match.match_semantic_id)
            # If we could not find the remote_matching_service, or we already checked it, we continue
            if remote_matching_service is None or remote_matching_service in already_checked_locations:
                # Note: There is an edge case where this does not find all matches:
                #   Imagine A -> B and C -> D are on SMR1 and B -> C is on SMR2. A query for A on SMR1 asks SMR2 for
                #   B, but SMR2 does not ask SMR1 for C, since SMR1 is already checked. Therefore, D is not found.
                continue

            # This makes it possible to create the match request:
            remote_matching_request = MatchRequest(
                semantic_id=match.match_semantic_id,
                # This is a simple inequality equation:
                #   Unified score is multiplied: score(A->B) * score(B->C)
                #   This score should be larger or equal than the requested score_limit:
                #   score(A->B) * score(B->C) >= score_limit
                #   score(A->B) is well known, as it is the `match.score`
                #   => score(B->C) >= (score_limit/score(A->B))
                score_limit=float(request_body.score_limit/match.score),
                # If we already request a remote score, it does not make sense to choose `local_only`
                local_only=False,
                already_checked_locations=already_checked_locations
            )
            url = f"{remote_matching_service}/query_matches"
            try:
                new_matches_response = requests.post(
                    url,
                    json=remote_matching_request.model_dump(mode="json"),
                    timeout=REMOTE_REQUEST_TIMEOUT
                )
                payload = new_matches_response.json()
            except (requests.RequestException, ValueError):
                continue  # An unreachable remote must not break the query, we still return all other matches
            if new_matches_response.status_code != 200 or not isinstance(payload, list):
                continue  # Protect against bad remotes by skipping them if they send nonsense
            for remote_match in (algorithm.SemanticMatch(**m) for m in payload):
                # The remote returns matches starting at `match.match_semantic_id`, so we need to prepend the local
                # part of the path: A -> B (local) and B -> C (remote) become A -> C.
                path = match.path + remote_match.path
                # Avoid paths that visit a `semantic_id` twice, including the ones going back to the queried one
                if len(set(path + [remote_match.match_semantic_id])) != len(path) + 1:
                    continue
                additional_remote_matches.append(algorithm.SemanticMatch(
                    base_semantic_id=request_body.semantic_id,
                    match_semantic_id=remote_match.match_semantic_id,
                    score=match.score * remote_match.score,  # path product
                    path=path,
                    metric_id=None,  # Multi-edge path: no single metric applies
                    graph_score=match.score * remote_match.score,
                ))
        # Finally, put all matches together and return
        for remote_match in additional_remote_matches:
            if remote_match not in matches:
                matches.append(remote_match)
        return matches

    def post_matches(
            self,
            request_body: List[algorithm.SemanticMatch]
    ) -> Response:
        for match in request_body:
            self.graph.add_semantic_match(
                base_semantic_id=match.base_semantic_id,
                match_semantic_id=match.match_semantic_id,
                score=float(match.score),
                metric_id=match.metric_id,
            )
        return Response(status_code=200)

    @staticmethod
    def _get_matcher_from_semantic_id(semantic_id: str) -> Optional[str]:
        """
        Finds the suiting `SemanticMatchRegistry` for the given `semantic_id`.

        :returns: The endpoint with which the `SemanticMatchRegistry` can be accessed
        """
        request_body = {"semantic_id": semantic_id}
        endpoint = config['RESOLVER']['endpoint']
        port = config['RESOLVER'].getint('port')
        url = f"{endpoint}:{port}/query_smr"
        try:
            response = requests.post(url, json=request_body, timeout=REMOTE_REQUEST_TIMEOUT)
        except requests.RequestException:
            return None  # Without a reachable discovery service, we can only return the local matches

        # Check if the response is successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response and construct SMSResponse object
            response_json = response.json()
            response_endpoint = response_json['smr_endpoint']
            return response_endpoint

        return None


if __name__ == '__main__':
    import os
    import configparser
    from fastapi import FastAPI
    import uvicorn

    config = configparser.ConfigParser()
    config.read([
        os.path.abspath(os.path.join(os.path.dirname(__file__), "./config.ini.default")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "./config.ini")),
    ])

    # Read in `SemanticMatchGraph`.
    # Note, this construct takes the path in the config.ini relative to the location of the config.ini
    match_graph = algorithm.SemanticMatchGraph.from_file(
        filename=os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..",
            config["SERVICE"]["match_graph_file"]
        ))
    )
    SEMANTIC_MATCHING_SERVICE = SemanticMatchRegistry(
        endpoint=config["SERVICE"]["endpoint"],
        graph=match_graph,
    )
    APP = FastAPI(
        title="Semantic Match Registry Service",
        description="tbd",
        contact={
            "name": "Sebastian Heppner",
            "url": "https://github.com/s-heppner",
        },
    )
    APP.include_router(
        SEMANTIC_MATCHING_SERVICE.router
    )
    uvicorn.run(APP, host=config["SERVICE"]["LISTEN_ADDRESS"], port=int(config["SERVICE"]["PORT"]))
