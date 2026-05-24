"""
Eval script: exercises the full recommendation loop for each sample candidate
without the browser. Run from the project root:

    python -m backend.eval

For each candidate it:
  1. Seeds preferences from their profile
  2. Gets initial top-3 recommendations
  3. Sends a canned feedback message
  4. Gets refined top-3
  5. Prints a preference diff and the two sets of job titles / scores
"""
from __future__ import annotations

import json
import copy

from .data import ALL_CANDIDATES
from .models import Preferences, MustConstraints
from .retriever import retrieve
from . import llm, prompts, sessions

# Canned feedback per candidate (keyed by name substring)
CANNED_FEEDBACK: dict[str, str] = {
    "Carin": "These feel too enterprise-heavy. I prefer earlier-stage startups and backend/AI roles. No enterprise SaaS.",
    "Scott": "I want more full-stack founding engineer roles at early-stage AI startups. Less research, more product.",
    "Warren": "I'd like more backend infrastructure / SRE roles at growth-stage companies. Less DevOps consulting.",
}

DEFAULT_FEEDBACK = "I prefer early-stage startups. Less enterprise, more backend engineering."


def _parse_preferences(raw: dict) -> Preferences:
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


def _get_feedback(name: str) -> str:
    for key, fb in CANNED_FEEDBACK.items():
        if key.lower() in name.lower():
            return fb
    return DEFAULT_FEEDBACK


def _rerank(candidate, prefs, label: str) -> list[dict]:
    retrieved, query, filters = retrieve(candidate, prefs)
    if not retrieved:
        print(f"  [!] No jobs matched hard filters for {label}")
        return []
    jobs_for_rerank = [r.job for r in retrieved]
    bm25_map = {r.job.id: r.bm25_score for r in retrieved}
    system, user = prompts.rerank_prompt(candidate, prefs, jobs_for_rerank)
    raw_ranks: list[dict] = llm.chat_json(system, user)  # type: ignore[assignment]
    job_map = {j.id: j for j in jobs_for_rerank}
    results = []
    for entry in raw_ranks[:3]:
        jid = str(entry.get("job_id", ""))
        job = job_map.get(jid)
        if not job:
            continue
        results.append({
            "title": job.title,
            "company": job.company,
            "bm25": round(bm25_map.get(jid, 0), 3),
            "rerank": entry.get("rerank_score", 0),
            "reasons": entry.get("reasons", [])[:2],
        })
    return results


def _pref_diff(before: Preferences, after: Preferences) -> dict:
    return {
        "prefer_added": [t for t in after.prefer if t not in before.prefer],
        "prefer_removed": [t for t in before.prefer if t not in after.prefer],
        "avoid_added": [t for t in after.avoid if t not in before.avoid],
        "remote_changed": before.must.remote_only != after.must.remote_only,
        "notes_after": after.free_text_notes,
    }


def run_eval():
    sep = "=" * 70
    for candidate in ALL_CANDIDATES:
        print(f"\n{sep}")
        print(f"CANDIDATE: {candidate.name}")
        print(f"Headline : {candidate.headline}")
        print(sep)

        # 1. Seed preferences
        print("\n[1/4] Seeding preferences from profile...")
        system, user = prompts.seed_preferences_prompt(candidate)
        raw_prefs: dict = llm.chat_json(system, user)  # type: ignore[assignment]
        prefs = _parse_preferences(raw_prefs)
        print(f"  prefer : {prefs.prefer}")
        print(f"  avoid  : {prefs.avoid}")
        print(f"  must   : {prefs.must.model_dump()}")
        print(f"  notes  : {prefs.free_text_notes}")

        # 2. Initial recommendations
        print("\n[2/4] Initial recommendations...")
        initial_jobs = _rerank(candidate, prefs, "initial")
        for i, j in enumerate(initial_jobs, 1):
            print(f"  #{i} [{j['rerank']}/100 | BM25={j['bm25']}] {j['title']} @ {j['company']}")
            for r in j["reasons"]:
                print(f"       • {r}")

        # 3. Send canned feedback
        feedback_msg = _get_feedback(candidate.name)
        print(f"\n[3/4] Feedback: \"{feedback_msg}\"")
        prefs_before = copy.deepcopy(prefs)
        system, user = prompts.update_preferences_prompt(
            prefs,
            # pass the initial job objects (we have titles/companies, good enough)
            [],  # last_jobs — prompts handles empty list gracefully
            feedback_msg,
        )
        raw_prefs2: dict = llm.chat_json(system, user)  # type: ignore[assignment]
        prefs_after = _parse_preferences(raw_prefs2)

        diff = _pref_diff(prefs_before, prefs_after)
        print(f"  Preference diff: {json.dumps(diff, indent=4)}")

        # 4. Refined recommendations
        print("\n[4/4] Refined recommendations...")
        refined_jobs = _rerank(candidate, prefs_after, "refined")
        for i, j in enumerate(refined_jobs, 1):
            print(f"  #{i} [{j['rerank']}/100 | BM25={j['bm25']}] {j['title']} @ {j['company']}")
            for r in j["reasons"]:
                print(f"       • {r}")

        # Check if recs actually changed
        initial_titles = {j["title"] for j in initial_jobs}
        refined_titles = {j["title"] for j in refined_jobs}
        overlap = initial_titles & refined_titles
        changed = len(initial_titles - refined_titles)
        print(f"\n  Result shift: {changed}/3 jobs changed after feedback")
        if overlap:
            print(f"  Still present: {overlap}")

    print(f"\n{sep}")
    print("Eval complete.")


if __name__ == "__main__":
    run_eval()
