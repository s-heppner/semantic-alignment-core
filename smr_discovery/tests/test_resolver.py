import unittest
from pathlib import Path
from typing import List

from unittest.mock import patch

from smr_discovery import resolver


FIXTURES = Path(__file__).parent / "test_resources"
ENDPOINTS_JSON = FIXTURES / "endpoints.json"


class TestResolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # sanity check: fixture exists
        assert ENDPOINTS_JSON.exists(), f"Missing test fixture: {ENDPOINTS_JSON}"

    def setUp(self):
        # load endpoints fresh for each test
        self.endpoints = resolver.SMREndpoints.from_file(str(ENDPOINTS_JSON))

    # ---------- helpers for DNS mocking ----------

    class _MockTXTRecord:
        def __init__(self, strings: List[bytes]):
            self.strings = strings

    def _mock_dns_with(self, *txt_values: str):
        """
        Returns a patcher that makes dns.resolver.resolve(domain, 'TXT') yield TXT records
        whose `.strings` lists contain the given values (as bytes).
        """
        records = [self._MockTXTRecord([s.encode("utf-8")]) for s in txt_values]
        return patch("smr_discovery.resolver.dns.resolver.resolve", return_value=records)

    def _mock_dns_noanswer(self):
        return patch("smr_discovery.resolver.dns.resolver.resolve", side_effect=resolver.dns.resolver.NoAnswer)

    def _mock_dns_nxdomain(self):
        return patch("smr_discovery.resolver.dns.resolver.resolve", side_effect=resolver.dns.resolver.NXDOMAIN)

    # ---------- matches_irdi / is_iri_not_irdi ----------

    def test_matches_irdi_true(self):
        self.assertTrue(resolver.matches_irdi("0112-0001#01-ACK323#7"))

    def test_matches_irdi_false(self):
        self.assertFalse(resolver.matches_irdi("https://example.org/x"))

    def test_is_iri_not_irdi(self):
        self.assertIs(resolver.is_iri_not_irdi("https://example.org/x"), True)
        self.assertIs(resolver.is_iri_not_irdi("0112-0001#01-ACK323#7"), False)
        self.assertIsNone(resolver.is_iri_not_irdi("definitely-not-an-irdi-or-iri"))

    # ---------- SMREndpoints.from_file ----------

    def test_from_file_loads_and_requires_fallback(self):
        ep = resolver.SMREndpoints.from_file(str(ENDPOINTS_JSON))
        self.assertIn("FALLBACK", ep._backend)
        self.assertEqual(ep._backend["FALLBACK"], "https://s-heppner.com/fallback_smr")

    # ---------- IRI resolution ----------

    def test_iri_backend_domain_hit(self):
        # domain present in JSON → direct hit, no DNS
        url = "https://s-heppner.com/some/path?x=1"
        got = self.endpoints.get_smr_from_semantic_id(url)
        self.assertEqual(got, "https://s-heppner.com/smr")

    def test_iri_dns_txt_hit(self):
        # domain not in JSON → use DNS TXT "smr: <url>"
        url = "https://example.org/whatever"
        with self._mock_dns_with("smr: https://dns.example.org/smr"):
            got = self.endpoints.get_smr_from_semantic_id(url)
        self.assertEqual(got, "https://dns.example.org/smr")

    def test_iri_dns_txt_ignored_then_fallback(self):
        # domain not in JSON; DNS has no smr TXT → fallback
        url = "https://no-such-domain.tld/foo"
        with self._mock_dns_noanswer():
            got = self.endpoints.get_smr_from_semantic_id(url)
        self.assertEqual(got, "https://s-heppner.com/fallback_smr")

    def test_iri_dns_nxdomain_then_fallback(self):
        url = "https://also-bad.tld/foo"
        with self._mock_dns_nxdomain():
            got = self.endpoints.get_smr_from_semantic_id(url)
        self.assertEqual(got, "https://s-heppner.com/fallback_smr")

    # ---------- IRDI resolution ----------

    def test_irdi_0112_maps_to_cdd(self):
        irdi = "0112-0001#01-ACK323#7"
        got = self.endpoints.get_smr_from_semantic_id(irdi)
        self.assertEqual(got, "https://s-heppner.com/cdd_smr")

    def test_irdi_0173_maps_to_eclass(self):
        irdi = "0173-0001#01-ACK323#7"
        got = self.endpoints.get_smr_from_semantic_id(irdi)
        self.assertEqual(got, "https://s-heppner.com/eclass_smr")

    def test_irdi_unknown_prefix_fallback(self):
        irdi = "9999-0001#01-ACK323#7"  # unknown ICD
        got = self.endpoints.get_smr_from_semantic_id(irdi)
        self.assertEqual(got, "https://s-heppner.com/fallback_smr")

    # ---------- neither IRI nor IRDI ----------

    def test_neither_returns_none(self):
        self.assertIsNone(self.endpoints.get_smr_from_semantic_id("not-a-url-or-irdi"))
