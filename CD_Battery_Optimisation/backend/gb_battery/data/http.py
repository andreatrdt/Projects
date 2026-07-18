"""Resilient HTTP client: retries, backoff, timeouts, polite rate limiting.

Wraps ``httpx`` with ``tenacity`` retry logic. Every successful fetch returns the
JSON payload plus the UTC retrieval timestamp so callers can record data lineage.
Raw payloads can be persisted for auditability.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gb_battery.data.settings import DataSettings, get_settings
from gb_battery.settlement import UTC


class DataSourceError(RuntimeError):
    """Raised when a data source cannot be reached or returns an error."""


class OfflineError(DataSourceError):
    """Raised when a live fetch is attempted in offline mode."""


_RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


class ResilientClient:
    """A thin, polite, retrying JSON HTTP client."""

    def __init__(self, settings: DataSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._last_request_at = 0.0
        self._lock = threading.Lock()
        self._client = httpx.Client(
            timeout=self.settings.request_timeout_s,
            headers={"User-Agent": self.settings.user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ResilientClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self.settings.min_request_interval_s - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> tuple[Any, datetime]:
        """GET ``url`` and return ``(payload, retrieved_at_utc)``.

        Retries transient failures with exponential backoff. Raises
        :class:`OfflineError` in offline mode and :class:`DataSourceError` on
        exhausted retries.
        """
        if self.settings.offline:
            raise OfflineError(f"Offline mode: refusing to fetch {url}")

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.settings.max_retries),
            wait=wait_exponential(multiplier=self.settings.backoff_base_s, max=8.0),
            retry=retry_if_exception_type(_RETRYABLE),
        )
        def _do() -> Any:
            self._throttle()
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

        try:
            payload = _do()
        except Exception as exc:  # noqa: BLE001 - normalise to DataSourceError
            raise DataSourceError(f"Failed to fetch {url}: {exc}") from exc
        return payload, datetime.now(tz=UTC)
