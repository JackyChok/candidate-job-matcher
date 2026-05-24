"""
Loads jobs.json and candidates.json once at startup and builds the BM25 index.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from .models import Candidate, Job

DATA_DIR = Path(__file__).parent.parent / "data"


def _normalise_job(raw: dict[str, Any], idx: int) -> Job:
    location = raw.get("location") or ""
    sponsorship = raw.get("sponsorship") or ""
    job_type = raw.get("job_type") or ""

    is_remote = bool(re.search(r"\bremote\b", location, re.I))
    will_sponsor = bool(re.search(r"will sponsor", sponsorship, re.I))

    return Job(
        id=str(idx),
        title=raw.get("title") or "",
        company=raw.get("company") or "",
        url=raw.get("url") or "",
        location=location,
        salary=raw.get("salary"),
        equity=raw.get("equity"),
        experience=raw.get("experience"),
        job_type=job_type,
        sponsorship=sponsorship,
        yc_batch=raw.get("yc_batch"),
        description=raw.get("description") or "",
        is_remote=is_remote,
        will_sponsor=will_sponsor,
    )


def _job_to_tokens(job: Job) -> list[str]:
    """Flatten key fields into a token list for BM25."""
    text = " ".join([
        job.title,
        job.company,
        job.location,
        job.job_type,
        job.experience or "",
        job.sponsorship,
        job.yc_batch or "",
        # first 1200 chars of description is plenty for BM25
        job.description[:1200],
    ])
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalise_candidate(raw: dict[str, Any]) -> Candidate:
    return Candidate(
        person_id=raw.get("person_id", 0),
        name=raw.get("name") or "",
        headline=raw.get("headline") or "",
        summary=raw.get("summary") or "",
        location=raw.get("location") or "",
        skills=raw.get("skills") or [],
        all_titles=raw.get("all_titles") or [],
        all_employers=raw.get("all_employers") or [],
        education_background=raw.get("education_background") or [],
        languages=raw.get("languages") or [],
        raw=raw,
    )


class JobIndex:
    """BM25 index over all jobs. Built once at import time."""

    def __init__(self, jobs: list[Job]) -> None:
        self.jobs = jobs
        self._corpus = [_job_to_tokens(j) for j in jobs]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, top_k: int = 30) -> list[tuple[Job, float]]:
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.jobs[i], float(s)) for i, s in ranked[:top_k] if s > 0]


def _load() -> tuple[list[Job], list[Candidate], JobIndex]:
    with open(DATA_DIR / "jobs.json", encoding="utf-8") as f:
        raw_jobs: list[dict] = json.load(f)

    with open(DATA_DIR / "candidates.json", encoding="utf-8") as f:
        raw_candidates: list[dict] = json.load(f)

    jobs = [_normalise_job(r, i) for i, r in enumerate(raw_jobs)]
    candidates = [_normalise_candidate(r) for r in raw_candidates]
    index = JobIndex(jobs)
    return jobs, candidates, index


# Module-level singletons – loaded once
ALL_JOBS, ALL_CANDIDATES, JOB_INDEX = _load()
