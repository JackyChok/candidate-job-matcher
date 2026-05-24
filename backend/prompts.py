"""
All LLM prompt functions.

Each function returns a (system, user) tuple for llm.chat / llm.chat_json.
"""
from __future__ import annotations

import json

from .models import Candidate, Job, Preferences


# ---------------------------------------------------------------------------
# 1. Seed preferences from a candidate profile
# ---------------------------------------------------------------------------

_SEED_SYSTEM = """\
You are a career advisor. Given a candidate's LinkedIn profile, extract their
job search preferences as structured JSON.

Return ONLY valid JSON — no markdown, no commentary.

Schema:
{
  "must": {
    "remote_only": false,
    "require_sponsorship": false,
    "job_types": [],
    "excluded_locations": []
  },
  "prefer": [],
  "avoid": [],
  "free_text_notes": ""
}

Guidelines:
- "prefer" should list 3-8 concise keywords describing ideal roles/companies
  (e.g. "early-stage startup", "backend", "AI/ML", "founding engineer", "B2B SaaS")
- "avoid" should list things the candidate is unlikely to want based on their profile
- "must.remote_only" = true only if their current location strongly suggests it
  or their title is remote-first
- Keep it grounded in the actual profile; do not invent preferences
- "free_text_notes" can be 1-2 sentences summarising the candidate's career stage
"""


def seed_preferences_prompt(candidate: Candidate) -> tuple[str, str]:
    profile_summary = json.dumps(
        {
            "name": candidate.name,
            "headline": candidate.headline,
            "summary": candidate.summary[:800],
            "location": candidate.location,
            "skills": candidate.skills[:20],
            "recent_titles": candidate.all_titles[:6],
            "recent_employers": candidate.all_employers[:6],
            "education": [
                {
                    "degree": e.get("degree_name"),
                    "school": e.get("institute_name"),
                    "field": e.get("field_of_study"),
                }
                for e in candidate.education_background[:3]
            ],
        },
        indent=2,
    )
    return _SEED_SYSTEM, f"Candidate profile:\n{profile_summary}"


# ---------------------------------------------------------------------------
# 2. Rerank jobs and generate reasons
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = """\
You are a job matching expert. Given a candidate profile, their current
preferences, and a list of candidate jobs, return the top 3 best matches.

Return ONLY valid JSON — no markdown, no commentary.

Schema:
[
  {
    "job_id": "<id string>",
    "rerank_score": <integer 0-100>,
    "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"],
    "concerns": ["<concern 1>"]
  },
  ...
]

Guidelines:
- rerank_score: 100 = perfect fit. Weight candidate-job skill overlap (40%),
  seniority match (20%), company stage/type match (20%), preference alignment (20%).
- reasons: 2-4 bullet-sized strings explaining why this job fits.
- concerns: 0-2 honest caveats (e.g. location mismatch, seniority stretch).
- Heavily penalise jobs that mention terms in `preferences.avoid`.
- Prefer jobs that mention terms in `preferences.prefer`.
- Return exactly 3 entries, ordered by rerank_score descending.
"""


def _compress_job(job: Job) -> dict:
    return {
        "id": job.id,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "salary": job.salary,
        "job_type": job.job_type,
        "experience": job.experience,
        "sponsorship": job.sponsorship,
        "yc_batch": job.yc_batch,
        "description_preview": job.description[:600],
    }


def rerank_prompt(
    candidate: Candidate,
    prefs: Preferences,
    jobs: list[Job],
) -> tuple[str, str]:
    payload = {
        "candidate": {
            "name": candidate.name,
            "headline": candidate.headline,
            "summary": candidate.summary[:400],
            "skills": candidate.skills[:20],
            "recent_titles": candidate.all_titles[:5],
            "location": candidate.location,
        },
        "preferences": prefs.model_dump(),
        "candidate_jobs": [_compress_job(j) for j in jobs],
    }
    return _RERANK_SYSTEM, json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# 3. Update preferences from feedback
# ---------------------------------------------------------------------------

_UPDATE_PREFS_SYSTEM = """\
You are a career advisor helping refine a candidate's job search preferences.

You will receive:
- The candidate's current preferences (JSON)
- The 3 jobs shown in the last round (titles + companies)
- The user's natural-language feedback about those results

Update the preferences to reflect the feedback and return ONLY the updated
preferences JSON — no markdown, no commentary.

Schema (same as input):
{
  "must": {
    "remote_only": false,
    "require_sponsorship": false,
    "job_types": [],
    "excluded_locations": []
  },
  "prefer": [],
  "avoid": [],
  "free_text_notes": ""
}

Guidelines:
- Add terms to "avoid" if feedback says results were too X ("too enterprise",
  "too full-stack", "too sales-heavy").
- Add terms to "prefer" if feedback expresses positive direction ("more backend",
  "earlier stage", "AI-focused").
- Set must.remote_only=true if feedback says "remote only".
- Update free_text_notes to capture the evolving preference in plain language.
- Be conservative: do not wipe out existing preferences unless explicitly contradicted.
"""


def update_preferences_prompt(
    prefs: Preferences,
    last_jobs: list[Job],
    feedback: str,
) -> tuple[str, str]:
    payload = {
        "current_preferences": prefs.model_dump(),
        "last_shown_jobs": [
            {"title": j.title, "company": j.company} for j in last_jobs
        ],
        "user_feedback": feedback,
    }
    return _UPDATE_PREFS_SYSTEM, json.dumps(payload, indent=2)
