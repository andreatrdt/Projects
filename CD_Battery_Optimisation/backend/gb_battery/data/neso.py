"""NESO (National Energy System Operator) data-portal adapter via the CKAN API.

Uses CKAN action endpoints to *discover* dataset resources (``package_search`` /
``package_show``) rather than hardcoding a single resource id, then pulls records
with ``datastore_search``. Relevant NESO datasets include demand data updates,
day-ahead demand/wind/solar forecasts, interconnector flows, daily balancing costs,
constraint volumes/costs, EAC auction results and the Dynamic Containment /
Moderation / Regulation and reserve products.

Resource ids drift over time, so discovery is the default path; a known resource id
can still be passed directly for speed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from gb_battery.data.http import DataSourceError, ResilientClient
from gb_battery.data.settings import DataSettings, get_settings

SOURCE = "neso"


class NesoClient:
    """Thin CKAN client for the NESO data portal."""

    def __init__(self, settings: DataSettings | None = None, client: ResilientClient | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client or ResilientClient(self.settings)

    def _action(self, action: str, params: dict[str, Any]) -> Any:
        url = f"{self.settings.neso_base_url}/{action}"
        payload, _ = self._client.get_json(url, params=params)
        if not isinstance(payload, dict) or not payload.get("success", False):
            raise DataSourceError(f"NESO CKAN action '{action}' failed")
        return payload["result"]

    def search_datasets(self, query: str, rows: int = 10) -> list[dict]:
        """Search datasets (packages) by free text; return lightweight descriptors."""
        result = self._action("package_search", {"q": query, "rows": rows})
        out = []
        for pkg in result.get("results", []):
            out.append(
                {
                    "name": pkg.get("name"),
                    "title": pkg.get("title"),
                    "id": pkg.get("id"),
                    "resources": [
                        {"id": r.get("id"), "name": r.get("name"), "format": r.get("format")}
                        for r in pkg.get("resources", [])
                    ],
                }
            )
        return out

    def find_resource(self, query: str, fmt: str = "CSV") -> str | None:
        """Return the first datastore-active resource id matching ``query``."""
        for pkg in self.search_datasets(query, rows=10):
            for res in pkg["resources"]:
                if res.get("format", "").upper() == fmt.upper() and res.get("id"):
                    return res["id"]
        return None

    def datastore_search(self, resource_id: str, limit: int = 1000, filters: dict | None = None) -> pd.DataFrame:
        """Pull records from a CKAN datastore resource into a DataFrame."""
        params: dict[str, Any] = {"resource_id": resource_id, "limit": limit}
        if filters:
            import json

            params["filters"] = json.dumps(filters)
        result = self._action("datastore_search", params)
        records = result.get("records", [])
        df = pd.DataFrame(records)
        df.attrs["resource_id"] = resource_id
        df.attrs["source"] = f"{SOURCE}.ckan.{resource_id}"
        return df
