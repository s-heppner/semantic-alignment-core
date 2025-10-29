from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from smr_discovery import resolver


class SMRRequest(BaseModel):
    semantic_id: str


class SMRResponse(BaseModel):
    smr_endpoint: str
    meta_information: Dict


class SMRDiscoveryService:
    """
    A Service, resolving semantic_ids to their respective Semantic Match Registry endpoint
    """
    def __init__(
            self,
            endpoint: str,
            smr_endpoints: resolver.SMREndpoints
    ):
        """
        Initializer of :class:`~.SemanticMatchingService`

        :ivar endpoint: The endpoint on which the service listens
        :ivar smr_endpoints: The :class:`resolver.SMREndpoints` object
        """
        self.router = APIRouter()
        self.router.add_api_route(
            "/get_smr",
            self.get_smr,
            methods=["GET"]
        )
        self.endpoint: str = endpoint
        self.smr_endpoints: resolver.SMREndpoints = smr_endpoints

    def get_smr(
            self,
            request_body: SMRRequest
    ) -> SMRResponse:
        """
        Returns a Semantic Matching Service for a given semantic_id
        """
        endpoint: Optional[str] = self.smr_endpoints.get_smr_from_semantic_id(request_body.semantic_id)
        if endpoint is None:
            raise HTTPException(
                status_code=404,
                detail=f"No Semantic Match Registry endpoint found for semantic_id='{request_body.semantic_id}'"
            )
        return SMRResponse(
            smr_endpoint=endpoint,
            meta_information={}
        )


if __name__ == '__main__':
    import os
    import argparse
    from fastapi import FastAPI
    import uvicorn

    parser = argparse.ArgumentParser(description="SMR Discovery Service")

    parser.add_argument(
        "--endpoints-json-file",
        type=str,
        default=os.getenv("SMR_ENDPOINTS_JSON_FILE", ""),
        help="Path to SMR endpoints JSON file (env: SMR_ENDPOINTS_JSON_FILE, REQUIRED)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("SMR_ENDPOINT", "http://127.0.0.1"),
        help="Public base URL of this service (env: SMR_ENDPOINT, default: http://127.0.0.1)"
    )
    parser.add_argument(
        "--listen-address",
        type=str,
        default=os.getenv("SMR_LISTEN_ADDRESS", "0.0.0.0"),
        help="Host/IP to bind (env: SMR_LISTEN_ADDRESS, default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SMR_PORT", "8125")),
        help="Port to bind (env: SMR_PORT, default: 8125)"
    )

    args = parser.parse_args()

    if not args.endpoints_json_file:
        raise SystemExit(
            "SMR endpoints file not provided. Set --endpoints-json-file "
            "or SMR_ENDPOINTS_JSON_FILE."
        )

    # Wire up service
    SMR_DISCOVERY_SERVICE = SMRDiscoveryService(
        endpoint=args.endpoint,
        smr_endpoints=resolver.SMREndpoints.from_file(args.endpoints_json_file),
    )

    APP = FastAPI()
    APP.include_router(SMR_DISCOVERY_SERVICE.router)

    uvicorn.run(APP, host=args.listen_address, port=args.port)
