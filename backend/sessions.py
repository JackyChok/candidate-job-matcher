"""
In-memory session store. No DB needed.
Each session holds a Candidate + evolving Preferences + full history.
"""
from __future__ import annotations

import uuid
from typing import Optional

from .models import SessionState, Candidate, Preferences

_SESSIONS: dict[str, SessionState] = {}


def create_session(candidate: Candidate, initial_preferences: Preferences) -> SessionState:
    sid = str(uuid.uuid4())
    state = SessionState(
        session_id=sid,
        candidate=candidate,
        preferences=initial_preferences,
    )
    _SESSIONS[sid] = state
    return state


def get_session(session_id: str) -> Optional[SessionState]:
    return _SESSIONS.get(session_id)


def save_session(state: SessionState) -> None:
    _SESSIONS[state.session_id] = state
