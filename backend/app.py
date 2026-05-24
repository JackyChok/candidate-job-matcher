"""
FastAPI application — all routes.
Also serves the frontend as static files at /.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .data import ALL_CANDIDATES, ALL_JOBS
from .llm import RateLimitError
from .models import (
    Candidate,
    HistoryEntry,
    MustConstraints,
    Preferences,
    RecommendedJob,
    RoundDebug,
    RoundResult,
)
from fastapi.responses import JSONResponse
from .retriever import retrieve
from . import llm, prompts, sessions

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="Candidate Job Matcher", version="1.0")


@app.exception_handler(RateLimitError)
def rate_limit_handler(request, exc: RateLimitError):
    return JSONResponse(status_code=429, content={"detail": str(exc)})

# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CandidateSummary(BaseModel):
    person_id: int
    name: str
    headline: str
    location: str


class CreateSessionRequest(BaseModel):
    candidate_id: Optional[int] = None       # person_id from /candidates
    candidate: Optional[dict[str, Any]] = None  # ad-hoc full JSON


class CreateSessionResponse(BaseModel):
    session_id: str
    candidate: CandidateSummary
    preferences: Preferences


class RecommendRequest(BaseModel):
    session_id: str


class FeedbackRequest(BaseModel):
    session_id: str
    feedback: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_preferences(raw: dict) -> Preferences:
    """Safely parse an LLM-returned preferences dict."""
    must_raw = raw.get("must", {})
    must = MustConstraints(
        remote_only=bool(must_raw.get("remote_only", False)),
        require_sponsorship=bool(must_raw.get("require_sponsorship", False)),
        job_types=list(must_raw.get("job_types") or []),
        excluded_locations=list(must_raw.get("excluded_locations") or []),
    )
    return Preferences(
        must=must,
        prefer=list(raw.get("prefer") or []),
        avoid=list(raw.get("avoid") or []),
        free_text_notes=str(raw.get("free_text_notes") or ""),
    )


def _run_recommend(state) -> RoundResult:
    """Core pipeline: retrieve → rerank → return RoundResult."""
    retrieved, query, filters = retrieve(state.candidate, state.preferences)

    if not retrieved:
        raise HTTPException(
            status_code=422,
            detail="No jobs matched the hard constraints. Try relaxing your filters.",
        )

    jobs_for_rerank = [r.job for r in retrieved]
    bm25_score_map = {r.job.id: r.bm25_score for r in retrieved}

    system, user = prompts.rerank_prompt(state.candidate, state.preferences, jobs_for_rerank)
    try:
        raw_ranks: list[dict] = llm.chat_json(system, user)  # type: ignore[assignment]
    except RateLimitError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Map job_id -> Job for fast lookup
    job_map = {j.id: j for j in jobs_for_rerank}

    recommendations: list[RecommendedJob] = []
    for entry in raw_ranks[:3]:
        jid = str(entry.get("job_id", ""))
        job = job_map.get(jid)
        if job is None:
            continue
        recommendations.append(
            RecommendedJob(
                job=job,
                bm25_score=round(bm25_score_map.get(jid, 0.0), 4),
                rerank_score=int(entry.get("rerank_score", 0)),
                reasons=list(entry.get("reasons") or []),
                concerns=list(entry.get("concerns") or []),
            )
        )

    state.round += 1
    return RoundResult(
        round=state.round,
        preferences=state.preferences,
        jobs=recommendations,
        debug=RoundDebug(
            query=query,
            retrieved_count=len(retrieved),
            after_filter_count=len(retrieved),
            filters_applied=filters,
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/candidates", response_model=list[CandidateSummary])
def list_candidates():
    return [
        CandidateSummary(
            person_id=c.person_id,
            name=c.name,
            headline=c.headline,
            location=c.location,
        )
        for c in ALL_CANDIDATES
    ]


@app.post("/session", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    # Resolve candidate
    if req.candidate is not None:
        # ad-hoc: parse raw JSON into Candidate
        from .data import _normalise_candidate
        candidate = _normalise_candidate(req.candidate)
    elif req.candidate_id is not None:
        matches = [c for c in ALL_CANDIDATES if c.person_id == req.candidate_id]
        if not matches:
            raise HTTPException(status_code=404, detail="Candidate not found")
        candidate = matches[0]
    else:
        raise HTTPException(status_code=400, detail="Provide candidate_id or candidate")

    # Seed preferences via LLM
    system, user = prompts.seed_preferences_prompt(candidate)
    try:
        raw_prefs: dict = llm.chat_json(system, user)  # type: ignore[assignment]
    except RateLimitError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    initial_prefs = _parse_preferences(raw_prefs)

    state = sessions.create_session(candidate, initial_prefs)

    return CreateSessionResponse(
        session_id=state.session_id,
        candidate=CandidateSummary(
            person_id=candidate.person_id,
            name=candidate.name,
            headline=candidate.headline,
            location=candidate.location,
        ),
        preferences=initial_prefs,
    )


@app.post("/recommend", response_model=RoundResult)
def recommend(req: RecommendRequest):
    state = sessions.get_session(req.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    prefs_before = copy.deepcopy(state.preferences)
    result = _run_recommend(state)

    state.history.append(
        HistoryEntry(
            round=result.round,
            feedback=None,
            preferences_before=prefs_before,
            preferences_after=state.preferences,
            result=result,
        )
    )
    sessions.save_session(state)
    return result


@app.post("/feedback", response_model=RoundResult)
def feedback(req: FeedbackRequest):
    state = sessions.get_session(req.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not state.history:
        raise HTTPException(status_code=400, detail="No recommendations yet. Call /recommend first.")

    last_jobs = [r.job for r in state.history[-1].result.jobs]
    prefs_before = copy.deepcopy(state.preferences)

    # Update preferences via LLM
    system, user = prompts.update_preferences_prompt(state.preferences, last_jobs, req.feedback)
    try:
        raw_prefs: dict = llm.chat_json(system, user)  # type: ignore[assignment]
    except RateLimitError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    state.preferences = _parse_preferences(raw_prefs)

    result = _run_recommend(state)

    state.history.append(
        HistoryEntry(
            round=result.round,
            feedback=req.feedback,
            preferences_before=prefs_before,
            preferences_after=state.preferences,
            result=result,
        )
    )
    sessions.save_session(state)
    return result


@app.get("/session/{session_id}")
def get_session_history(session_id: str):
    state = sessions.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.model_dump()
