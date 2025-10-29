import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from smr_discovery.service import SMRDiscoveryService
from smr_discovery import resolver


class TestSMRService(unittest.TestCase):
    """Integration-style unit tests for the /get_smr endpoint."""

    @classmethod
    def setUpClass(cls):
        # Resolve fixture path once
        cls.fixtures = Path(__file__).parent / "test_resources"
        cls.endpoints_json = cls.fixtures / "endpoints.json"
        assert cls.endpoints_json.exists(), f"Missing fixture: {cls.endpoints_json}"

        # Wire the real resolver with fixture endpoints
        endpoints = resolver.SMREndpoints.from_file(str(cls.endpoints_json))
        svc = SMRDiscoveryService(
            endpoint="http://localhost:8125",
            smr_endpoints=endpoints,
        )
        app = FastAPI()
        app.include_router(svc.router)

        # One client for the whole class
        cls.client = TestClient(app)

    # ---------- helpers ----------

    class _MockTXT:
        def __init__(self, s: str):
            # dnspython exposes .strings as a list[bytes]
            self.strings = [s.encode("utf-8")]

    def _patch_dns_txt(self, *txt_values: str):
        """Patch dns.resolver.resolve to return TXT records with provided string values."""
        records = [self._MockTXT(v) for v in txt_values]
        return patch("smr_discovery.resolver.dns.resolver.resolve", return_value=records)

    def _patch_dns_noanswer(self):
        return patch(
            "smr_discovery.resolver.dns.resolver.resolve",
            side_effect=resolver.dns.resolver.NoAnswer,
        )

    def _patch_dns_nxdomain(self):
        return patch(
            "smr_discovery.resolver.dns.resolver.resolve",
            side_effect=resolver.dns.resolver.NXDOMAIN,
        )

    # ---------- tests ----------

    def test_get_smr_iri_backend_match(self):
        r = self.client.request("GET", "/get_smr", json={"semantic_id": "https://s-heppner.com/whatever"})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["smr_endpoint"], "https://s-heppner.com/smr")
        self.assertIsInstance(data["meta_information"], dict)

    def test_get_smr_dns_txt_fallback(self):
        # Domain not in endpoints.json; DNS TXT provides smr: URL
        with self._patch_dns_txt("smr: https://dns.example.org/smr"):
            r = self.client.request("GET", "/get_smr", json={"semantic_id": "https://no-entry.example/path"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["smr_endpoint"], "https://dns.example.org/smr")

    def test_get_smr_dns_noanswer_uses_fallback(self):
        with self._patch_dns_noanswer():
            r = self.client.request("GET", "/get_smr", json={"semantic_id": "https://no-such-dns.example/x"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["smr_endpoint"], "https://s-heppner.com/fallback_smr")

    def test_get_smr_dns_nxdomain_uses_fallback(self):
        with self._patch_dns_nxdomain():
            r = self.client.request("GET", "/get_smr", json={"semantic_id": "https://still-bad.example/x"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["smr_endpoint"], "https://s-heppner.com/fallback_smr")

    def test_get_smr_irdi_0112_maps_to_cdd(self):
        r = self.client.request("GET", "/get_smr", json={"semantic_id": "0112-0001#01-ACK323#7"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["smr_endpoint"], "https://s-heppner.com/cdd_smr")

    def test_get_smr_irdi_0173_maps_to_eclass(self):
        r = self.client.request("GET", "/get_smr", json={"semantic_id": "0173-0001#01-ACK323#7"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["smr_endpoint"], "https://s-heppner.com/eclass_smr")

    def test_get_smr_neither_irdi_nor_iri_returns_404(self):
        r = self.client.request("GET", "/get_smr", json={"semantic_id": "not-a-url-or-irdi"})
        self.assertEqual(r.status_code, 404)
        self.assertIn("No Semantic Match Registry endpoint found", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
