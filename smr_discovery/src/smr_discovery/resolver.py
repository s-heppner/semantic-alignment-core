import dataclasses
from typing import Optional, Dict
from urllib.parse import urlparse
import json
import re

import dns.resolver


def matches_irdi(s: str) -> bool:
    # (2024-09-11, s-heppner)
    # This pattern stems from the wonderful IRDI-Parser project:
    # https://github.com/moritzsommer/irdi-parser
    # Sadly, we had problems with Docker installing and finding the package, so we decided to eliminate the dependency.
    irdi_pattern = re.compile(
        # International Code Designator (4 digits)
        r'^(?P<icd>\d{4})-'
        # Organization Identifier (4 safe characters)
        r'(?P<org_id>[a-zA-Z0-9]{4})'
        # Optional Additional Information (4 safe characters)
        r'(-(?P<add_info>[a-zA-Z0-9]{4}))?'
        # Separator Character
        r'#'
        # Code Space Identifier (2 safe characters)
        r'(?P<csi>[a-zA-Z0-9]{2})-'
        # Item Code (6 safe characters)
        r'(?P<item_code>[a-zA-Z0-9]{6})'
        # Separator Character
        r'#'
        # Version Identifier (1 digit)
        r'(?P<version>\d)$'
    )
    return bool(irdi_pattern.match(s))


def is_iri_not_irdi(semantic_id: str) -> Optional[bool]:
    """
    :return: `True`, if `semantic_id` is an IRI, False if it is an IRDI, None for neither
    """
    # Check IRDI
    if matches_irdi(semantic_id):
        return False
    # Check IRI
    parsed_url = urlparse(semantic_id)
    if parsed_url.scheme:
        return True
    # Not IRDI or IRI
    return None


def _get_smr_dns_record(domain: str) -> Optional[str]:
    """
    Returns the `smr` DNS TXT record for the given domain, if it exists.
    """
    try:
        result = dns.resolver.resolve(domain, 'TXT')
        for txt_record in result:
            if txt_record.strings and txt_record.strings[0].decode().startswith("smr"):
                smr_dns_record = txt_record.strings[0].decode()
                try:
                    endpoint = smr_dns_record.split(": ")[-1]
                    return endpoint
                except Exception as e:
                    print(f"Cannot parse TXT record {smr_dns_record} for {domain}: {e}")
                    return None
        print(f"No DNS TXT record starting with 'smr' found for {domain}")
        return None
    except dns.resolver.NXDOMAIN:
        print(f"No DNS records found for {domain}")
        return None
    except dns.resolver.NoAnswer:
        print(f"No TXT records found for {domain}")
        return None


@dataclasses.dataclass
class SMREndpoints:
    _backend: Dict[str, str]

    @classmethod
    def from_file(cls, filename: str) -> "SMREndpoints":
        """
        Parse an `endpoints.json` file.

        The file needs to be a JSON dict with the following entry types:

        ```JSON
        {
            "FALLBACK": "<endpoint>",
            "<international code designator, if IRDI>": "<endpoint>",
            "<domain_name, if IRI>": "<endpoint>",
        }
        ```
        """
        with open(filename, "r") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(f"Endpoints file must be a JSON object, got {type(data).__name__}")

        # We require a FALLBACK endpoint
        if not data.get("FALLBACK"):
            raise KeyError(f"{filename} misses required endpoint 'FALLBACK'")

        return SMREndpoints(_backend=data)

    def get_smr_from_semantic_id(self, semantic_id: str) -> Optional[str]:
        """
        Returns the suiting SMR endpoint to the given semantic_id, if it exists in the _backend.

        This function always does the following:
         - First, we try to see if we have the specific semantic_id in the backend
         - Then we check, if there is a DNS entry for the semantic_id (only for IRIs)
         - Lastly we take the fallback endpoint, if it exists
        """
        # IRI
        if is_iri_not_irdi(semantic_id) is True:
            # First, try to see if the semantic_id is in the backend
            domain = urlparse(semantic_id).netloc  # Parse the given semantic_id and just use the main domain name
            if self._backend.get(domain):
                return self._backend.get(domain)

            # Second, try to find the "smr" DNS TXT record
            dns_endpoint: Optional[str] = _get_smr_dns_record(domain)
            if dns_endpoint is not None:
                return dns_endpoint

            # Third, check the `FALLBACK` option
            return self._backend.get("FALLBACK")

        # IRDI
        elif is_iri_not_irdi(semantic_id) is False:
            # First, see if the semanticID is in the endpoint:
            international_code_designator = semantic_id[:4]  # First 4 digits o the semantic_id
            if self._backend.get(international_code_designator):
                return self._backend.get(international_code_designator)

            # Otherwise, we can only try to get the fallback endpoint
            return self._backend.get("FALLBACK")

        # Nothing we can do
        else:
            return None
