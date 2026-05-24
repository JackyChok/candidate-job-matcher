"""
Retrieval layer: BM25 search + hard-constraint filtering.

Step 1: build a query string from candidate profile + current preferences
Step 2: BM25 returns top-30
Step 3: apply hard filters from Preferences.must -> keep up to 20
"""
from __future__ import annotations

import re
from typing import NamedTuple

from .data import JOB_INDEX
from .models import Candidate, Job, Preferences


class RetrievedJob(NamedTuple):
    job: Job
    bm25_score: float


def build_query(candidate: Candidate, prefs: Preferences) -> str:
    """
    Construct a keyword query that steers BM25 toward what we want.
    Prefer-terms are included; avoid-terms are intentionally excluded (BM25
    has no native negation—the rerank LLM handles avoidance on its side).
    """
    parts: list[str] = []

    # core candidate signal
    if candidate.headline:
        parts.append(candidate.headline)
    parts.extend(candidate.skills[:15])
    parts.extend(candidate.all_titles[:5])

    # positive preferences
    parts.extend(prefs.prefer)

    # add 'remote' if must.remote_only
    if prefs.must.remote_only:
        parts.append("remote")

    # add preferred job types
    parts.extend(prefs.must.job_types)

    return " ".join(parts)


def _passes_hard_filters(job: Job, prefs: Preferences) -> tuple[bool, list[str]]:
    """Return (passes, list_of_applied_filter_names_that_excluded)."""
    filters_hit: list[str] = []

    if prefs.must.remote_only and not job.is_remote:
        filters_hit.append("remote_only")

    if prefs.must.require_sponsorship and not job.will_sponsor:
        filters_hit.append("require_sponsorship")

    if prefs.must.job_types:
        normalised_types = [t.lower() for t in prefs.must.job_types]
        if job.job_type.lower() not in normalised_types:
            filters_hit.append(f"job_type:{prefs.must.job_types}")

    if prefs.must.excluded_locations:
        for excl in prefs.must.excluded_locations:
            if re.search(re.escape(excl), job.location, re.I):
                filters_hit.append(f"excluded_location:{excl}")
                break

    return len(filters_hit) == 0, filters_hit


def retrieve(
    candidate: Candidate,
    prefs: Preferences,
    bm25_top_k: int = 30,
    max_after_filter: int = 20,
) -> tuple[list[RetrievedJob], str, list[str]]:
    """
    Returns:
        retrieved     – up to `max_after_filter` jobs that pass hard filters
        query         – the BM25 query string used (for debug)
        filters_used  – list of filter names that were active
    """
    query = build_query(candidate, prefs)
    raw_results = JOB_INDEX.search(query, top_k=bm25_top_k)

    # collect which hard filters are active (for debug panel)
    active_filters: list[str] = []
    if prefs.must.remote_only:
        active_filters.append("remote_only")
    if prefs.must.require_sponsorship:
        active_filters.append("require_sponsorship")
    if prefs.must.job_types:
        active_filters.append(f"job_type:{prefs.must.job_types}")
    if prefs.must.excluded_locations:
        active_filters.append(f"excluded_locations:{prefs.must.excluded_locations}")

    retrieved: list[RetrievedJob] = []
    for job, score in raw_results:
        passes, _ = _passes_hard_filters(job, prefs)
        if passes:
            retrieved.append(RetrievedJob(job=job, bm25_score=score))
        if len(retrieved) >= max_after_filter:
            break

    return retrieved, query, active_filters
