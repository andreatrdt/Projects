"""In-memory replay session registry.

Sessions are ephemeral research artefacts (a replay is cheap to recreate), so a
bounded in-process dict is deliberate: no database schema, no cleanup jobs.
Oldest sessions are evicted once the cap is reached.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from gb_battery.replay.engine import ReplayEngine

MAX_SESSIONS = 40


@dataclass
class ReplaySession:
    replay_id: str
    engine: ReplayEngine
    mode: str  # "historical" | "live"
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


class SessionRegistry:
    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[str, ReplaySession] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def create(self, engine: ReplayEngine, mode: str) -> ReplaySession:
        session = ReplaySession(replay_id=uuid.uuid4().hex[:12], engine=engine, mode=mode)
        with self._lock:
            self._sessions[session.replay_id] = session
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
        return session

    def get(self, replay_id: str) -> ReplaySession | None:
        with self._lock:
            return self._sessions.get(replay_id)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)


REGISTRY = SessionRegistry()
