"""Local Parquet cache with DuckDB metadata for incremental, resilient ingestion.

Each cached table is a Parquet file keyed by ``<source>__<key>.parquet``. A DuckDB
database records ingestion metadata (source, key, retrieved_at, rows) so the UI can
show the last successful update per source and warn on stale data.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb
import pandas as pd

from gb_battery.data.settings import DataSettings, get_settings
from gb_battery.settlement import UTC


@dataclass
class CacheEntry:
    source: str
    key: str
    retrieved_at: datetime
    rows: int
    path: str

    def is_stale(self, stale_after_hours: float, now: datetime | None = None) -> bool:
        now = now or datetime.now(tz=UTC)
        return (now - self.retrieved_at) > timedelta(hours=stale_after_hours)


class ParquetCache:
    """Parquet table store + DuckDB metadata index."""

    def __init__(self, settings: DataSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dir = self.settings.ensure_cache_dir()
        self._meta_path = self.dir / "_metadata.duckdb"
        self._init_meta()

    def _init_meta(self) -> None:
        con = duckdb.connect(str(self._meta_path))
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_metadata (
                source VARCHAR,
                key VARCHAR,
                retrieved_at TIMESTAMPTZ,
                rows INTEGER,
                path VARCHAR,
                PRIMARY KEY (source, key)
            )
            """
        )
        con.close()

    def _file(self, source: str, key: str) -> str:
        safe = f"{source}__{key}".replace("/", "_").replace(":", "-")
        return str(self.dir / f"{safe}.parquet")

    def put(self, source: str, key: str, df: pd.DataFrame, retrieved_at: datetime | None = None) -> CacheEntry:
        """Best-effort cache write.

        The Parquet payload is the important artefact; the DuckDB metadata upsert is
        best-effort and must never raise into the caller. Under concurrent identical
        writes DuckDB's MVCC can raise a commit conflict — we retry once, then swallow,
        so a caching hiccup never discards freshly-fetched live data.
        """
        retrieved_at = retrieved_at or datetime.now(tz=UTC)
        path = self._file(source, key)
        with contextlib.suppress(Exception):  # caching is best-effort
            df.to_parquet(path, index=False)
        for attempt in range(2):
            try:
                con = duckdb.connect(str(self._meta_path))
                try:
                    con.execute(
                        """
                        INSERT INTO ingest_metadata (source, key, retrieved_at, rows, path)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (source, key) DO UPDATE SET
                            retrieved_at = excluded.retrieved_at,
                            rows = excluded.rows,
                            path = excluded.path
                        """,
                        [source, key, retrieved_at, len(df), path],
                    )
                finally:
                    con.close()
                break
            except Exception:  # noqa: BLE001 - retry once then give up
                if attempt == 1:
                    break
        return CacheEntry(source, key, retrieved_at, len(df), path)

    def get(self, source: str, key: str) -> pd.DataFrame | None:
        entry = self.entry(source, key)
        if entry is None:
            return None
        try:
            return pd.read_parquet(entry.path)
        except FileNotFoundError:
            return None

    def entry(self, source: str, key: str) -> CacheEntry | None:
        con = duckdb.connect(str(self._meta_path))
        row = con.execute(
            "SELECT source, key, retrieved_at, rows, path FROM ingest_metadata WHERE source=? AND key=?",
            [source, key],
        ).fetchone()
        con.close()
        if row is None:
            return None
        return CacheEntry(row[0], row[1], row[2], int(row[3]), row[4])

    def last_update(self, source: str) -> datetime | None:
        con = duckdb.connect(str(self._meta_path))
        row = con.execute(
            "SELECT max(retrieved_at) FROM ingest_metadata WHERE source=?", [source]
        ).fetchone()
        con.close()
        return row[0] if row and row[0] is not None else None

    def all_entries(self) -> list[CacheEntry]:
        con = duckdb.connect(str(self._meta_path))
        rows = con.execute(
            "SELECT source, key, retrieved_at, rows, path FROM ingest_metadata ORDER BY source, key"
        ).fetchall()
        con.close()
        return [CacheEntry(r[0], r[1], r[2], int(r[3]), r[4]) for r in rows]
