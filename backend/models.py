from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class Job(BaseModel):
    id: str  # synthetic: index in jobs.json as string
    title: str
    company: str
    url: str = ""
    location: str = ""
    salary: Optional[str] = None
    equity: Optional[str] = None
    experience: Optional[str] = None
    job_type: str = ""
    sponsorship: str = ""
    yc_batch: Optional[str] = None
    description: str = ""
    # normalised flags derived at load time
    is_remote: bool = False
    will_sponsor: bool = False


class Candidate(BaseModel):
    person_id: int
    name: str
    headline: str = ""
    summary: str = ""
    location: str = ""
    skills: list[str] = Field(default_factory=list)
    all_titles: list[str] = Field(default_factory=list)
    all_employers: list[str] = Field(default_factory=list)
    education_background: list[dict[str, Any]] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    # raw linkedin data preserved so LLM can read it
    raw: dict[str, Any] = Field(default_factory=dict)


class MustConstraints(BaseModel):
    remote_only: bool = False
    require_sponsorship: bool = False
    job_types: list[str] = Field(default_factory=list)   # e.g. ["Full-time"]
    excluded_locations: list[str] = Field(default_factory=list)


class Preferences(BaseModel):
    must: MustConstraints = Field(default_factory=MustConstraints)
    prefer: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    free_text_notes: str = ""


class RecommendedJob(BaseModel):
    job: Job
    bm25_score: float
    rerank_score: int  # 0-100 from LLM
    reasons: list[str]
    concerns: list[str]


class RoundDebug(BaseModel):
    query: str
    retrieved_count: int
    after_filter_count: int
    filters_applied: list[str]


class RoundResult(BaseModel):
    round: int
    preferences: Preferences
    jobs: list[RecommendedJob]
    debug: RoundDebug


class HistoryEntry(BaseModel):
    round: int
    feedback: Optional[str]          # None for round 1 (initial)
    preferences_before: Preferences
    preferences_after: Preferences
    result: RoundResult


class SessionState(BaseModel):
    session_id: str
    candidate: Candidate
    preferences: Preferences = Field(default_factory=Preferences)
    history: list[HistoryEntry] = Field(default_factory=list)
    round: int = 0
