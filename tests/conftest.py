"""
Shared pytest fixtures.
"""
from __future__ import annotations

import pytest

from backend.models import Candidate, Job, MustConstraints, Preferences


@pytest.fixture
def sample_candidate() -> Candidate:
    return Candidate(
        person_id=42,
        name="Test Candidate",
        headline="Senior Backend Engineer, ex-Stripe, looking for early-stage AI startups",
        location="San Francisco, CA",
        summary="6 years building distributed systems at scale.",
        skills=["Python", "Go", "AWS", "Kafka", "distributed systems"],
        all_titles=["Senior Backend Engineer", "Software Engineer"],
        all_employers=["Stripe", "Google"],
    )


@pytest.fixture
def sample_jobs() -> list[Job]:
    """A small handful of fake jobs spanning different traits."""
    return [
        Job(
            id="1",
            title="Founding Engineer at AI Startup",
            company="EarlyAI",
            location="San Francisco, CA, US / Remote",
            job_type="Full-time",
            sponsorship="Will sponsor",
            yc_batch="W25",
            description="Build distributed systems for our LLM platform. Python, Go, AWS, Kafka.",
            is_remote=True,
            will_sponsor=True,
        ),
        Job(
            id="2",
            title="Senior Backend Engineer at MegaCorp",
            company="MegaCorp Inc",
            location="New York, NY, US",
            job_type="Full-time",
            sponsorship="US citizen/visa only",
            description="Enterprise SaaS for Fortune 500. Java, Spring, Oracle.",
            is_remote=False,
            will_sponsor=False,
        ),
        Job(
            id="3",
            title="Pokemon Content Intern",
            company="Misprint",
            location="New York, NY, US",
            job_type="Internship",
            sponsorship="US citizen/visa only",
            description="Make Pokemon TikTok content. Marketing focus.",
            is_remote=False,
            will_sponsor=False,
        ),
        Job(
            id="4",
            title="Founding Backend Engineer (Remote)",
            company="Seedling",
            location="Remote",
            job_type="Full-time",
            sponsorship="Will sponsor",
            yc_batch="S25",
            description="Series A AI infrastructure. Python, distributed systems, AWS.",
            is_remote=True,
            will_sponsor=True,
        ),
    ]


@pytest.fixture
def empty_prefs() -> Preferences:
    return Preferences()


@pytest.fixture
def backend_prefs() -> Preferences:
    return Preferences(
        must=MustConstraints(),
        prefer=["backend", "early-stage startup", "AI"],
        avoid=["enterprise"],
        free_text_notes="Senior backend engineer seeking AI startups",
    )


@pytest.fixture
def remote_prefs() -> Preferences:
    return Preferences(
        must=MustConstraints(remote_only=True, require_sponsorship=True),
        prefer=["backend", "AI"],
        avoid=[],
    )
