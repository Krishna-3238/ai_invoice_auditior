"""Route invoice questions and derive safe, explicit metadata filters."""

import re
from typing import Any, Dict


class QueryRouterAgent:
    """Extract exact invoice/vendor identifiers without guessing from free text."""

    INVOICE_PATTERNS = (
        re.compile(r"\b(INV[-_ ]?[A-Z0-9][A-Z0-9_-]*)\b", re.IGNORECASE),
        re.compile(r"\binvoice\s+(?:number\s*)?[#:]?\s*([A-Z0-9][A-Z0-9_-]*)\b", re.IGNORECASE),
    )

    def route(self, query: str) -> Dict[str, Any]:
        filters: Dict[str, str] = {}
        for pattern in self.INVOICE_PATTERNS:
            match = pattern.search(query)
            if match:
                raw = re.sub(r"\s+", "-", match.group(1).strip()).upper()
                if raw.startswith("INV"):
                    invoice_no = "INV-" + raw[3:].lstrip("-_")
                else:
                    invoice_no = raw
                filters["invoice_no"] = invoice_no
                break

        vendor_match = re.search(
            r"\bvendor(?:\s+(?:id|code)\s*(?:is|=|:)?|\s*:\s*)([A-Z0-9][A-Z0-9_-]*)\b",
            query,
            re.IGNORECASE,
        )
        if vendor_match:
            filters["vendor_id"] = vendor_match.group(1).upper()

        route = "invoice_lookup" if "invoice_no" in filters else (
            "vendor_lookup" if "vendor_id" in filters else "semantic_search"
        )
        return {"route": route, "filters": filters, "query": query}
