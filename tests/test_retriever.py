"""
Tests for the retrieval layer: query construction + hard filtering.
"""
from __future__ import annotations

import pytest

from backend.models import MustConstraints, Preferences
from backend.retriever import _passes_hard_filters, build_query, retrieve


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def test_query_includes_headline(sample_candidate, empty_prefs):
    q = build_query(sample_candidate, empty_prefs)
    assert "Senior Backend Engineer" in q


def test_query_includes_skills(sample_candidate, empty_prefs):
    q = build_query(sample_candidate, empty_prefs)
    assert "Python" in q
    assert "Go" in q


def test_query_includes_prefer_terms(sample_candidate, backend_prefs):
    q = build_query(sample_candidate, backend_prefs)
    assert "backend" in q
    assert "early-stage startup" in q


def test_query_adds_remote_when_remote_only(sample_candidate, remote_prefs):
    q = build_query(sample_candidate, remote_prefs)
    assert "remote" in q.lower()


def test_query_omits_remote_when_not_required(sample_candidate, empty_prefs):
    q = build_query(sample_candidate, empty_prefs)
    assert "remote" not in q.lower()


def test_query_includes_job_types(sample_candidate):
    prefs = Preferences(must=MustConstraints(job_types=["Full-time"]))
    q = build_query(sample_candidate, prefs)
    assert "Full-time" in q


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------


def test_no_filters_lets_everything_pass(sample_jobs, empty_prefs):
    for job in sample_jobs:
        passes, hits = _passes_hard_filters(job, empty_prefs)
        assert passes is True
        assert hits == []


def test_remote_only_filter_drops_onsite(sample_jobs):
    prefs = Preferences(must=MustConstraints(remote_only=True))
    onsite = [j for j in sample_jobs if not j.is_remote][0]
    passes, hits = _passes_hard_filters(onsite, prefs)
    assert passes is False
    assert "remote_only" in hits


def test_remote_only_filter_keeps_remote(sample_jobs):
    prefs = Preferences(must=MustConstraints(remote_only=True))
    remote = [j for j in sample_jobs if j.is_remote][0]
    passes, _ = _passes_hard_filters(remote, prefs)
    assert passes is True


def test_sponsorship_filter_drops_non_sponsor(sample_jobs):
    prefs = Preferences(must=MustConstraints(require_sponsorship=True))
    non_sponsor = [j for j in sample_jobs if not j.will_sponsor][0]
    passes, hits = _passes_hard_filters(non_sponsor, prefs)
    assert passes is False
    assert "require_sponsorship" in hits


def test_job_type_filter_drops_mismatch(sample_jobs):
    prefs = Preferences(must=MustConstraints(job_types=["Full-time"]))
    internship = [j for j in sample_jobs if j.job_type == "Internship"][0]
    passes, hits = _passes_hard_filters(internship, prefs)
    assert passes is False
    assert any("job_type" in h for h in hits)


def test_job_type_filter_keeps_match(sample_jobs):
    prefs = Preferences(must=MustConstraints(job_types=["Full-time"]))
    full_time = [j for j in sample_jobs if j.job_type == "Full-time"][0]
    passes, _ = _passes_hard_filters(full_time, prefs)
    assert passes is True


def test_excluded_location_filter(sample_jobs):
    prefs = Preferences(must=MustConstraints(excluded_locations=["New York"]))
    ny_job = [j for j in sample_jobs if "New York" in j.location][0]
    passes, hits = _passes_hard_filters(ny_job, prefs)
    assert passes is False
    assert any("excluded_location" in h for h in hits)


def test_multiple_filters_combine(sample_jobs):
    prefs = Preferences(must=MustConstraints(remote_only=True, require_sponsorship=True))
    # MegaCorp is neither remote nor sponsoring — should hit both
    megacorp = [j for j in sample_jobs if j.company == "MegaCorp Inc"][0]
    passes, hits = _passes_hard_filters(megacorp, prefs)
    assert passes is False
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# Full retrieve()  (uses real BM25 index)
# ---------------------------------------------------------------------------


def test_retrieve_returns_results(sample_candidate, empty_prefs):
    results, query, filters = retrieve(sample_candidate, empty_prefs)
    assert len(results) > 0
    assert "Senior Backend Engineer" in query
    assert filters == []


def test_retrieve_respects_max_after_filter(sample_candidate, empty_prefs):
    results, _, _ = retrieve(sample_candidate, empty_prefs, max_after_filter=5)
    assert len(results) <= 5


def test_retrieve_reports_active_filters(sample_candidate):
    prefs = Preferences(must=MustConstraints(remote_only=True))
    _, _, filters = retrieve(sample_candidate, prefs)
    assert "remote_only" in filters


def test_retrieve_results_have_positive_scores(sample_candidate, empty_prefs):
    results, _, _ = retrieve(sample_candidate, empty_prefs)
    for r in results:
        assert r.bm25_score > 0
