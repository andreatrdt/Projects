"""Runtime settings for data access (env-overridable)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cache_dir() -> Path:
    # Repo-root/data/cache — kept out of git; frozen samples live in the package.
    return Path(__file__).resolve().parents[3] / "data" / "cache"


class DataSettings(BaseSettings):
    """Configuration for external data adapters and caching.

    All fields can be overridden with ``GBB_`` env vars, e.g. ``GBB_OFFLINE=1``.
    """

    model_config = SettingsConfigDict(env_prefix="GBB_", extra="ignore")

    elexon_base_url: str = "https://data.elexon.co.uk/bmrs/api/v1"
    neso_base_url: str = "https://api.neso.energy/api/3/action"

    user_agent: str = "gb-battery-coopt/0.1 (+research; contact via GitHub)"
    request_timeout_s: float = 25.0
    max_retries: int = 4
    backoff_base_s: float = 0.5
    min_request_interval_s: float = 0.2  # polite rate limiting

    # If True, adapters never hit the network and serve frozen/cached data only.
    offline: bool = False

    cache_dir: Path = _default_cache_dir()

    # Hours after which cached data is considered stale (for UI warnings).
    stale_after_hours: float = 6.0

    def ensure_cache_dir(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir


_settings: DataSettings | None = None


def get_settings() -> DataSettings:
    global _settings
    if _settings is None:
        _settings = DataSettings()
    return _settings
