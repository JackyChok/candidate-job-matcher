"""
End-to-end API tests using FastAPI's TestClient.
The LLM is mocked so tests run offline and deterministically.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.llm import RateLimitError


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Canned LLM responses
# ---------------------------------------------------------------------------

CANNED_PREFS = {
    "must": {
        "remote_only": False,
        "require_sponsorship": False,
        "job_types": [],
        "excluded_locations": [],
    },
    "prefer": ["backend", "early-stage startup"],
    "avoid": ["enterprise"],
    "free_text_notes": "Senior backend engineer seeking startups",
}

UPDATED_PREFS = {
    "must": {
        "remote_only": True,
        "require_sponsorship": False,
        "job_types": [],
        "excluded_locations": [],
    },
    "prefer": ["backend", "early-stage startup", "AI"],
    "avoid": ["enterprise", "operations"],
    "free_text_notes": "Now also remote-only",
}


def _make_rerank_response(user_payload: str) -> list[dict]:
    """
    Extract the first 3 real job ids from the rerank prompt's user payload
    so the canned response uses ids that actually exist in the retrieved set.
    """
    data = json.loads(user_payload)
    job_ids = [j["id"] for j in data["candidate_jobs"][:3]]
    return [
        {"job_id": job_ids[0], "rerank_score": 92, "reasons": ["Skill overlap"], "concerns": []},
        {"job_id": job_ids[1], "rerank_score": 85, "reasons": ["Cultural fit"], "concerns": ["X"]},
        {"job_id": job_ids[2], "rerank_score": 78, "reasons": ["Founding role"], "concerns": []},
    ]


def _llm_responses():
    """
    Fake llm.chat_json that returns:
      1st call → canned seed preferences
      2nd call → rerank response using real retrieved job ids
      3rd call → updated preferences
      4th call → rerank response using real retrieved job ids
    """
    state = {"call": 0}

    def fake(system, user):
        state["call"] += 1
        n = state["call"]
        if n == 1:
            return CANNED_PREFS
        if n == 2:
            return _make_rerank_response(user)
        if n == 3:
            return UPDATED_PREFS
        return _make_rerank_response(user)

    return fake


# ---------------------------------------------------------------------------
# /candidates
# ---------------------------------------------------------------------------


def test_list_candidates(client):
    resp = client.get("/candidates")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    names = {c["name"] for c in data}
    assert "Carin Gan" in names


# ---------------------------------------------------------------------------
# /session
# ---------------------------------------------------------------------------


def test_create_session_with_candidate_id(client):
    with patch("backend.app.llm.chat_json", side_effect=_llm_responses()):
        resp = client.post("/session", json={"candidate_id": 2847095})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["candidate"]["name"] == "Carin Gan"
    assert "prefer" in data["preferences"]


def test_create_session_with_adhoc_candidate(client):
    payload = {
        "candidate": {
            "person_id": 999,
            "name": "Ad Hoc",
            "headline": "Engineer",
            "skills": ["Python"],
        }
    }
    with patch("backend.app.llm.chat_json", side_effect=_llm_responses()):
        resp = client.post("/session", json=payload)
    assert resp.status_code == 200
    assert resp.json()["candidate"]["name"] == "Ad Hoc"


def test_create_session_unknown_candidate_id(client):
    with patch("backend.app.llm.chat_json", side_effect=_llm_responses()):
        resp = client.post("/session", json={"candidate_id": 999999999})
    assert resp.status_code == 404


def test_create_session_missing_input(client):
    resp = client.post("/session", json={})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------


def test_recommend_returns_top_3(client):
    with patch("backend.app.llm.chat_json", side_effect=_llm_responses()):
        session_resp = client.post("/session", json={"candidate_id": 2847095})
        sid = session_resp.json()["session_id"]
        rec_resp = client.post("/recommend", json={"session_id": sid})

    assert rec_resp.status_code == 200
    data = rec_resp.json()
    assert data["round"] == 1
    assert len(data["jobs"]) == 3
    for j in data["jobs"]:
        assert "bm25_score" in j
        assert "rerank_score" in j
        assert "reasons" in j


def test_recommend_unknown_session(client):
    resp = client.post("/recommend", json={"session_id": "does-not-exist"})
    assert resp.status_code == 404


def test_recommend_debug_includes_query(client):
    with patch("backend.app.llm.chat_json", side_effect=_llm_responses()):
        sid = client.post("/session", json={"candidate_id": 2847095}).json()["session_id"]
        data = client.post("/recommend", json={"session_id": sid}).json()

    assert "query" in data["debug"]
    assert data["debug"]["retrieved_count"] > 0


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------


def test_feedback_updates_preferences_and_returns_round_2(client):
    with patch("backend.app.llm.chat_json", side_effect=_llm_responses()):
        sid = client.post("/session", json={"candidate_id": 2847095}).json()["session_id"]
        client.post("/recommend", json={"session_id": sid})
        fb_resp = client.post("/feedback", json={"session_id": sid, "feedback": "remote only please"})

    assert fb_resp.status_code == 200
    data = fb_resp.json()
    assert data["round"] == 2
    assert data["preferences"]["must"]["remote_only"] is True
    assert "AI" in data["preferences"]["prefer"]


def test_feedback_without_prior_recommendation_errors(client):
    with patch("backend.app.llm.chat_json", side_effect=_llm_responses()):
        sid = client.post("/session", json={"candidate_id": 2847095}).json()["session_id"]
        # skip /recommend; go straight to /feedback
        resp = client.post("/feedback", json={"session_id": sid, "feedback": "x"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /session/{id}
# ---------------------------------------------------------------------------


def test_get_session_history(client):
    with patch("backend.app.llm.chat_json", side_effect=_llm_responses()):
        sid = client.post("/session", json={"candidate_id": 2847095}).json()["session_id"]
        client.post("/recommend", json={"session_id": sid})
        client.post("/feedback", json={"session_id": sid, "feedback": "more AI"})

    resp = client.get(f"/session/{sid}")
    assert resp.status_code == 200
    state = resp.json()
    assert state["round"] == 2
    assert len(state["history"]) == 2


# ---------------------------------------------------------------------------
# Rate limit handling
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429(client):
    def raise_rate_limit(system, user):
        raise RateLimitError("Provider rate-limited")

    with patch("backend.app.llm.chat_json", side_effect=raise_rate_limit):
        resp = client.post("/session", json={"candidate_id": 2847095})

    assert resp.status_code == 429
    assert "rate" in resp.json()["detail"].lower()
