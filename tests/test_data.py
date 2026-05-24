"""
Tests for data loading + normalisation + BM25 index construction.
"""
from __future__ import annotations

import pytest

from backend.data import (
    ALL_CANDIDATES,
    ALL_JOBS,
    JOB_INDEX,
    _normalise_candidate,
    _normalise_job,
    _job_to_tokens,
)


# ---------------------------------------------------------------------------
# Data loaded at startup
# ---------------------------------------------------------------------------


def test_jobs_loaded():
    assert len(ALL_JOBS) > 1000, "Should load 1000+ jobs from jobs.json"


def test_candidates_loaded():
    assert len(ALL_CANDIDATES) == 3, "Three sample candidates expected"
    names = {c.name for c in ALL_CANDIDATES}
    assert "Carin Gan" in names


def test_bm25_index_built():
    assert JOB_INDEX is not None
    assert len(JOB_INDEX.jobs) == len(ALL_JOBS)


# ---------------------------------------------------------------------------
# Job normalisation
# ---------------------------------------------------------------------------


def test_remote_flag_detected():
    job = _normalise_job(
        {"title": "X", "company": "Y", "location": "San Francisco, CA / Remote"}, 0
    )
    assert job.is_remote is True


def test_remote_flag_negative():
    job = _normalise_job({"title": "X", "company": "Y", "location": "New York, NY"}, 0)
    assert job.is_remote is False


def test_sponsorship_flag_detected():
    job = _normalise_job(
        {"title": "X", "company": "Y", "sponsorship": "Will sponsor"}, 0
    )
    assert job.will_sponsor is True


def test_sponsorship_flag_negative():
    job = _normalise_job(
        {"title": "X", "company": "Y", "sponsorship": "US citizen/visa only"}, 0
    )
    assert job.will_sponsor is False


def test_job_id_is_string_index():
    job = _normalise_job({"title": "X", "company": "Y"}, 42)
    assert job.id == "42"


def test_missing_fields_default_gracefully():
    job = _normalise_job({}, 0)
    assert job.title == ""
    assert job.company == ""
    assert job.salary is None
    assert job.is_remote is False
    assert job.will_sponsor is False


# ---------------------------------------------------------------------------
# Candidate normalisation
# ---------------------------------------------------------------------------


def test_candidate_minimal_input():
    """A candidate with only a name and headline still parses."""
    c = _normalise_candidate({"name": "Jane", "headline": "Engineer"})
    assert c.name == "Jane"
    assert c.headline == "Engineer"
    assert c.skills == []
    assert c.person_id == 0


def test_candidate_full_input():
    c = _normalise_candidate({
        "person_id": 123,
        "name": "Jane",
        "headline": "Backend Engineer",
        "skills": ["Python", "Go"],
        "all_titles": ["Engineer"],
        "all_employers": ["Stripe"],
        "summary": "Lorem ipsum",
        "location": "NYC",
    })
    assert c.person_id == 123
    assert c.skills == ["Python", "Go"]
    assert c.location == "NYC"


def test_candidate_raw_preserved():
    raw = {"name": "Jane", "headline": "Eng", "custom_field": "value"}
    c = _normalise_candidate(raw)
    assert c.raw["custom_field"] == "value"


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokens_lowercased():
    job = _normalise_job({"title": "Backend Engineer", "company": "TechCo"}, 0)
    tokens = _job_to_tokens(job)
    assert "backend" in tokens
    assert "Backend" not in tokens


def test_tokens_split_on_punctuation():
    job = _normalise_job({"title": "Senior/Staff Engineer", "company": "Co"}, 0)
    tokens = _job_to_tokens(job)
    assert "senior" in tokens and "staff" in tokens


# ---------------------------------------------------------------------------
# BM25 search behaviour
# ---------------------------------------------------------------------------


def test_bm25_returns_results_for_common_query():
    results = JOB_INDEX.search("backend engineer Python", top_k=10)
    assert len(results) > 0
    # Each result is (Job, score)
    for job, score in results:
        assert score > 0


def test_bm25_empty_query_returns_nothing():
    results = JOB_INDEX.search("", top_k=10)
    assert results == []


def test_bm25_top_k_honoured():
    results = JOB_INDEX.search("engineer", top_k=5)
    assert len(results) <= 5


def test_bm25_results_sorted_descending():
    results = JOB_INDEX.search("backend Python engineer", top_k=20)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
