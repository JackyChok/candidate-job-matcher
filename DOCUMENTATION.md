# Candidate Job Matcher — Engineering Documentation

A complete walkthrough of what this system does and how it works.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [System Architecture](#2-system-architecture)
3. [Request Lifecycle (End-to-End)](#3-request-lifecycle-end-to-end)
4. [File-by-File Walkthrough](#4-file-by-file-walkthrough)
5. [Why These Tools / Tech Choices](#5-why-these-tools--tech-choices)
6. [Deep Dive: BM25 Retrieval](#6-deep-dive-bm25-retrieval)
7. [Deep Dive: LLM Rerank + Preference State](#7-deep-dive-llm-rerank--preference-state)
8. [Testing](#8-testing)
9. [Known Limitations](#9-known-limitations)
10. [Future Improvements](#10-future-improvements)

---

## 1. What This System Does

The system implements an **iterative job recommendation loop**:

1. User provides a candidate profile (LinkedIn-style JSON).
2. System returns the **top 3 best-matching jobs** from a dataset of 1,045 YC startup jobs, with reasons for each match.
3. User gives **natural-language feedback** ("too enterprise", "more backend", "remote only").
4. System updates its internal model of the candidate's preferences.
5. System returns refined recommendations.
6. Loop continues for as many rounds as needed.

The key challenge is **adapting recommendations meaningfully across rounds**, not just shuffling the same results.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Browser (vanilla JS + Tailwind)                                    │
│  ─ candidate picker, job cards, feedback textarea, prefs panel      │
└─────────────────────────────────────────────────────────────────────┘
                              │ HTTP (JSON)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI backend  (backend/app.py)                                  │
│  Routes: /candidates, /session, /recommend, /feedback, /session/:id │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐
│  In-memory   │    │  BM25 Retriever  │    │  LLM Client    │
│  Session     │    │  (rank-bm25)     │    │  (httpx)       │
│  Store       │    │                  │    │                │
│              │    │  1,045 jobs      │    │  OpenAI /      │
│  dict[id ->  │    │  tokenised at    │    │  Gemini /      │
│   State]     │    │  startup         │    │  Groq          │
└──────────────┘    └──────────────────┘    └────────────────┘
```

### The Two-Stage Pipeline

This is the heart of the system. Every recommendation goes through:

```
┌────────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   1,045 jobs       │ →  │  BM25 search     │ →  │  Top 30 jobs   │
│   (all)            │    │  (keyword)       │    │  (candidates)  │
└────────────────────┘    └──────────────────┘    └────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │  Hard filters         │
                                              │  (remote, sponsor,    │
                                              │   job type)           │
                                              └──────────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │  Top 20 candidates    │
                                              └──────────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │  LLM rerank           │
                                              │  (with candidate +    │
                                              │   preferences)        │
                                              └──────────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │  Top 3 with reasons   │
                                              │  + concerns + scores  │
                                              └──────────────────────┘
```

**Why two stages instead of "just ask the LLM"?**
- Sending 1,045 job descriptions to the LLM = expensive and slow
- BM25 narrows to ~20 in milliseconds, no API cost
- LLM only sees jobs that are already plausibly relevant
- This is the standard "retrieval + rerank" pattern used in production search systems

---

## 3. Request Lifecycle (End-to-End)

### Round 1: Initial Recommendation

```
User clicks "Start Session"
  │
  ▼
POST /session  { candidate_id: 2847095 }
  │
  ├─→ Look up candidate from ALL_CANDIDATES
  ├─→ Call LLM with seed_preferences_prompt()
  │   ↓ LLM returns initial Preferences JSON
  ├─→ Create SessionState in _SESSIONS dict
  └─→ Return { session_id, candidate, preferences }

User clicks "Get Recommendations"
  │
  ▼
POST /recommend  { session_id }
  │
  ├─→ Load SessionState
  ├─→ build_query(candidate, preferences)
  │   ↓ "Matching cofounders ... CoffeeSpace ... early-stage startup ..."
  ├─→ BM25.search(query, top_k=30)
  │   ↓ 30 jobs ranked by keyword overlap
  ├─→ Apply hard filters (remote, sponsorship, etc.)
  │   ↓ ~20 jobs survive
  ├─→ Call LLM with rerank_prompt()
  │   ↓ LLM returns [{job_id, rerank_score, reasons, concerns}, ...]
  ├─→ Save round to state.history
  └─→ Return { jobs[3], preferences, debug }
```

### Round 2+: Feedback Refinement

```
User types feedback "more backend, less operations" + clicks Refine
  │
  ▼
POST /feedback  { session_id, feedback }
  │
  ├─→ Load SessionState (still in memory)
  ├─→ Call LLM with update_preferences_prompt()
  │   ↓ Input: current Preferences + last 3 jobs + feedback
  │   ↓ Output: updated Preferences JSON
  ├─→ Replace state.preferences with the new one
  ├─→ Re-run the entire recommend pipeline (BM25 → filters → rerank)
  ├─→ Append to state.history
  └─→ Return refined { jobs[3], preferences, debug }
```

**Key insight:** the LLM is responsible for *interpreting* the feedback into structured preferences, not for the final ranking directly. This means the preference state is auditable JSON, not a fuzzy embedding.

---

## 4. File-by-File Walkthrough

### `backend/data.py` — Data loading + BM25 index

**Runs once at server startup.** Loads `jobs.json` and `candidates.json`, normalises every job into a `Job` Pydantic model, and builds a BM25 index over the full job corpus.

Key functions:
- `_normalise_job(raw, idx)` — pulls fields out of the raw JSON, derives flags like `is_remote` and `will_sponsor` from regex matches on the location/sponsorship strings.
- `_job_to_tokens(job)` — flattens each job into a list of lowercase tokens (title + company + location + first 1200 chars of description). This is what BM25 indexes.
- `JobIndex.search(query, top_k)` — tokenises the query, scores all jobs, returns the top N with positive scores.

Module-level singletons: `ALL_JOBS`, `ALL_CANDIDATES`, `JOB_INDEX` — loaded once and reused for every request.

### `backend/models.py` — Pydantic data classes

Defines the entire type system:
- `Job` — a single job posting (title, company, description, derived flags)
- `Candidate` — a LinkedIn-style profile (headline, skills, employers, etc.)
- `MustConstraints` — hard filters (remote_only, require_sponsorship, job_types, excluded_locations)
- `Preferences` — `must` + soft `prefer[]` + soft `avoid[]` + `free_text_notes`
- `RecommendedJob` — a job + its bm25_score + rerank_score + reasons[] + concerns[]
- `RoundResult` — the full response (round number, prefs, jobs, debug)
- `HistoryEntry` — captures feedback + prefs-before + prefs-after for each round
- `SessionState` — top-level container per user session

Using Pydantic gives us free input validation, automatic OpenAPI/Swagger docs, and structured JSON serialisation.

### `backend/retriever.py` — BM25 query + hard filtering

`retrieve(candidate, prefs)` does three things:
1. **Build a query string** from the candidate's headline, skills, titles, plus the soft `prefer` terms from preferences. Adds "remote" if `must.remote_only` is true.
2. **BM25 search** — returns top 30 by keyword overlap score.
3. **Hard filter** — drops any job that violates `must.*` constraints (remote, sponsorship, job type, excluded locations). Keeps up to 20.

Returns `(retrieved_jobs, query_string, active_filters)` — the query string and filter list are passed back so the UI can show retrieval debug info.

### `backend/llm.py` — Provider-agnostic LLM client

A thin `httpx` wrapper around three providers (Gemini, OpenAI, Groq) chosen by the `LLM_PROVIDER` env var. The same `chat(system, user)` interface works for all three.

Key details:
- `load_dotenv(override=True)` — reads `.env` at startup
- `_provider()` is a **function**, not a module constant, so the provider can be changed without restarting (just hot-reload). Same for the API keys.
- `chat()` retries once on a 429 with a 5-second wait, then raises `RateLimitError` so the caller can return a clean HTTP 429 instead of hanging.
- `chat_json()` wraps `chat()`, strips markdown fences (some models like to wrap JSON in ```json blocks), and parses to Python.

### `backend/prompts.py` — All LLM prompt functions

Three prompt builders. Each returns `(system_message, user_message)`.

**`seed_preferences_prompt(candidate)`**
> "Given this LinkedIn profile, extract job search preferences as JSON."

Used once at session start. Schema-locked: returns the same `Preferences` shape we use everywhere.

**`rerank_prompt(candidate, prefs, jobs)`**
> "Score and rank these 20 jobs for this candidate. Return top 3 with reasons."

Compresses each job to its key fields + first 600 chars of description to keep token usage down. Tells the LLM the scoring rubric:
- 40% skill overlap
- 20% seniority match
- 20% company stage/type
- 20% preference alignment

Also instructs it to **heavily penalise jobs that mention `prefs.avoid` terms**, since BM25 has no notion of negation.

**`update_preferences_prompt(prefs, last_jobs, feedback)`**
> "Here's the current prefs + last 3 jobs + user feedback. Return updated prefs JSON."

The feedback interpreter. Adds to `prefer` when feedback is positive, adds to `avoid` when feedback is negative, flips `must.remote_only` when feedback explicitly says "remote only". Conservative — doesn't wipe existing prefs unless explicitly contradicted.

### `backend/sessions.py` — In-memory session store

20 lines. A `dict[str, SessionState]`. `create_session()`, `get_session()`, `save_session()`. That's it.

Why no DB? Brief explicitly says no deployment/infra/CRUD. A demo session that survives until the server restart is sufficient.

### `backend/app.py` — FastAPI routes

The HTTP layer. Five routes:

| Route | Purpose |
|---|---|
| `GET /` | Serves `frontend/index.html` |
| `GET /candidates` | Lists the 3 sample candidates |
| `POST /session` | Creates a session; accepts either `{candidate_id}` or `{candidate: <full JSON>}` |
| `POST /recommend` | Runs the recommend pipeline; appends to history |
| `POST /feedback` | Updates prefs from feedback, then re-runs recommend |
| `GET /session/{id}` | Returns full session state (for debugging) |

Custom exception handler for `RateLimitError` returns a clean HTTP 429 with a useful message instead of a 500 stack trace.

`_run_recommend(state)` is the shared core used by both `/recommend` and `/feedback`.

### `backend/eval.py` — CLI sanity script

Runs the full loop (seed → recommend → feedback → re-recommend) for each of the 3 sample candidates and prints the results to stdout. Useful for:
- Verifying that feedback actually shifts results (it reports "Result shift: X/3 jobs changed")
- Demoing the system without the UI in the follow-up call
- Quick smoke testing after changes

### `frontend/index.html` + `frontend/app.js`

Single-page UI, vanilla JS, Tailwind from CDN. No build step, no framework.

Three panels:
- **Left** — candidate picker + paste JSON textarea + active profile card
- **Center** — empty state → controls → top-3 job cards → feedback textarea
- **Right** — live `Preferences` chips + retrieval debug + feedback history with diffs

The `app.js` is ~350 lines of straightforward DOM manipulation. All state is held in three module-level variables: `sessionId`, `currentPrefs`, `historyEntries`.

---

## 5. Why These Tools / Tech Choices

### Python + FastAPI

**Pros:**
- Most AI/ML libraries are Python-first (rank-bm25, embeddings, LLM SDKs)
- FastAPI is fast (built on Starlette + Pydantic), gives us free OpenAPI docs at `/docs`
- Type-hinted, async-ready, modern

**Trade-off considered:** Node.js + Express would also work, but Python wins because anything more sophisticated (embeddings, vector DBs) is easier to add later.

### Vanilla JS + Tailwind CDN

**Pros:**
- Zero build pipeline → faster setup, fewer moving parts
- The brief says "we are not evaluating frontend polish"
- Stays under 400 lines total

**Trade-off considered:** React would be cleaner for a real product, but adds Vite/webpack, node_modules, build steps. Not worth it for a demo.

### `rank-bm25` for retrieval (not embeddings)

This is the biggest design decision. See [Section 6](#6-deep-dive-bm25-retrieval) for the full reasoning.

**Short version:** Job postings are keyword-rich (Python, AWS, Founding Engineer, Backend, YC W25). BM25 handles this superbly with zero API cost and sub-millisecond queries. Embeddings would add latency, cost, and complexity for marginal quality gains on this domain.

### LLM for **rerank + preference state** (not direct ranking)

The LLM does two things only:
1. **Convert unstructured input → structured Preferences** (candidate profile → JSON, user feedback → updated JSON)
2. **Rank 20 jobs and explain why**

It never sees all 1,045 jobs. It never holds the preference state in its own context. The preferences live in our own typed Pydantic object, so we can inspect, log, persist, and debug them.

This is the right division of labour: deterministic systems (BM25, filters, storage) handle the boring scaling concerns; the LLM handles the parts that need natural-language understanding.

### `httpx` instead of provider SDKs

`openai`, `google-generativeai`, and `groq` all ship full SDKs, but each is megabytes and pulls in lots of transitive dependencies. The REST APIs of all three are simple JSON POSTs — `httpx` is 100KB and handles all of them with the same code. Keeps the dependency footprint tiny.

### In-memory dict for session storage

Brief says no DB. Demo sessions don't need durability. Zero setup. If this needed to scale, swap to Redis (1-line change).

### Pydantic v2 everywhere

Free input validation, automatic JSON schema, FastAPI integration. Less defensive code in our routes.

---

## 6. Deep Dive: BM25 Retrieval

### What is BM25?

**BM25 = Best Matching 25**, a probabilistic ranking function from information retrieval. It scores how well a query matches a document based on:

1. **Term frequency (TF)** — how often each query term appears in the document
2. **Inverse document frequency (IDF)** — rare terms (e.g. "Solidity") score higher than common terms (e.g. "the")
3. **Length normalisation** — long documents don't get unfairly boosted just for having more words

It's the same algorithm Elasticsearch uses by default. Battle-tested since 1994.

### Why BM25 for this problem?

| Property of our data | BM25 strength |
|---|---|
| Job descriptions are keyword-dense | BM25 excels at keyword matching |
| Candidate skills are concrete tokens (Python, AWS) | Exact token match is what we want |
| Dataset is small-medium (~1k docs) | BM25 scales trivially to millions |
| No GPU available | BM25 needs no model, just an index |
| Need fast iteration | Sub-millisecond queries |

### Why NOT embeddings?

Embeddings (e.g. OpenAI `text-embedding-3-small`, Sentence-BERT) would give us **semantic similarity** — "ML engineer" matching "machine learning role" even without word overlap.

Theoretical wins:
- Catches synonym matches BM25 misses
- Handles paraphrasing

Practical costs:
- Need to embed all 1,045 jobs upfront (~$0.02 once, but adds a build step)
- Need to embed every query (latency + cost per request)
- Need a vector index (FAISS / Chroma / pgvector)
- Need to re-embed if you change the model

For this dataset, the wins are marginal because:
- Job postings already use the standard industry terminology
- Candidates list skills using the same terminology
- The LLM rerank stage already handles semantic understanding on the 20 finalists

**Verdict:** embeddings add a lot of infra for a small quality gain. BM25 + LLM rerank is the right shape. If we did add embeddings, the smart move would be a *hybrid* (BM25 score + cosine similarity, combined) — but that's a future improvement, not v1.

### What our BM25 implementation does specifically

```python
# At startup:
for each job:
    tokens = lowercase + alphanumeric split of (title + company + location +
                                                job_type + experience + sponsorship +
                                                yc_batch + first 1200 chars of description)
BM25Okapi(corpus_of_tokens)

# At query time:
query_text = candidate.headline + skills[:15] + recent_titles[:5] + prefs.prefer
            + ("remote" if must.remote_only)
            + must.job_types
score every job → return top 30
```

Some interesting choices:
- We truncate descriptions to 1200 chars in the index (most jobs have key info in the first paragraph)
- We don't index the company description (too generic, would dilute scores)
- We don't lowercase / stem aggressively; the default tokeniser is fine for English job text

---

## 7. Deep Dive: LLM Rerank + Preference State

### The structured Preferences object

```json
{
  "must": {
    "remote_only": false,
    "require_sponsorship": false,
    "job_types": [],
    "excluded_locations": []
  },
  "prefer": ["backend", "early-stage startup", "AI/ML"],
  "avoid": ["enterprise", "full-stack"],
  "free_text_notes": "Founding-engineer level roles at seed/series-A AI companies"
}
```

This is the single source of truth for what the candidate wants. It drives:
- The BM25 query (`prefer` terms get appended)
- The hard filters (`must.*` becomes post-retrieval filters)
- The rerank prompt (the LLM is told to penalise `avoid`, reward `prefer`)
- The UI right panel (rendered as chips so the user can see what the system thinks)
- The feedback history (diffs are computed between rounds)

### Why structured instead of free-text?

You could imagine an alternative where we just keep a growing string like:
> "Wants backend roles, dislikes enterprise, prefers SF, doesn't want fullstack..."

…and append feedback to it. But:
- **Not inspectable** — user can't see at a glance what the system thinks
- **Not auditable** — preferences can silently drift
- **Hard to filter on** — can't extract "remote_only" cleanly for hard filtering
- **Token bloat** — grows unbounded with each round

Structured prefs solve all of this. The LLM is good at parsing free-form input → structured output; we leverage that and then operate on the structure deterministically.

### Two-tier preference model

| Tier | How it's applied | Examples |
|---|---|---|
| **`must.*`** (hard) | Post-BM25 filter — drops jobs entirely | "remote only", "needs visa sponsorship" |
| **`prefer[]`** (soft) | Boosts BM25 query + told to rerank-LLM | "backend", "early-stage", "AI" |
| **`avoid[]`** (soft) | Told to rerank-LLM (BM25 has no negation) | "enterprise", "operations" |

This separation matters because:
- Hard filters are deterministic and cheap
- Soft preferences benefit from the LLM's nuanced judgement
- The user can't accidentally filter out everything by being too specific

### Why the LLM rerank prompt works

The prompt explicitly tells the LLM:
1. The scoring rubric (40% skills / 20% seniority / 20% stage / 20% prefs)
2. To return exactly 3 results
3. To return JSON only (no markdown, no commentary)
4. To explain reasons and concerns

This gives consistent, parseable output. We also lower the temperature to 0.2 for stability.

If the LLM occasionally returns wrapped markdown (`""json ... """""`), `chat_json()` strips it before parsing.

---

## 8. Testing

The project ships with a **59-test pytest suite** covering data loading, retrieval, the LLM client, and the full API surface. Tests run **offline in under a second** because all LLM calls are mocked — no API key required to run them.

### Running the tests

```bash
# from the project root
.venv/bin/pytest

# or
python -m pytest
```

Expected output:

```
============================== 59 passed in 0.54s ==============================
```

### Test layout

```
tests/
  conftest.py          # shared fixtures (sample candidate, jobs, prefs)
  test_data.py         # data loading + normalisation + BM25 index (22 tests)
  test_retriever.py    # query construction + hard filters (18 tests)
  test_llm.py          # JSON parsing, markdown stripping, retry logic (11 tests)
  test_api.py          # full HTTP routes with mocked LLM (12 tests)
pytest.ini             # discovery + display config
```

### What each test file covers

**`test_data.py` — data layer**
- 1,045 jobs and 3 candidates load from JSON
- BM25 index builds successfully
- `is_remote` / `will_sponsor` flags derived correctly from regex matches
- Missing fields default gracefully (no crashes on partial JSON)
- Tokeniser is case-insensitive and splits on punctuation
- BM25 search returns sorted positive-score results

**`test_retriever.py` — retrieval logic**
- Query string includes candidate headline, skills, prefer terms
- `must.remote_only` adds "remote" to the query
- Each hard filter (`remote_only`, `require_sponsorship`, `job_types`, `excluded_locations`) drops the right jobs
- Multiple filters combine correctly (a job failing two reports both)
- `retrieve()` respects `max_after_filter` and reports active filters in debug output

**`test_llm.py` — LLM client**
- `chat_json` strips both ` ```json ` and plain ` ``` ` markdown fences
- Raises `JSONDecodeError` on truly invalid output (no silent failures)
- `chat()` retries once on 429, then raises `RateLimitError` (not a generic 500)
- Provider dispatch picks the right backend based on `LLM_PROVIDER` env var
- Non-429 errors (400, etc.) bubble up immediately without retry

**`test_api.py` — end-to-end HTTP**
- `GET /candidates` returns the three sample candidates
- `POST /session` works with both `candidate_id` and ad-hoc `candidate` JSON
- Unknown candidate_id returns 404
- `POST /recommend` returns exactly 3 jobs with `bm25_score`, `rerank_score`, `reasons`
- `POST /feedback` updates preferences (e.g. `must.remote_only` flips to true) and bumps round number
- Calling `/feedback` before `/recommend` returns a 400
- `RateLimitError` from the LLM client is correctly translated to an HTTP 429

### How LLM calls are mocked

API tests patch `backend.app.llm.chat_json` with a function that returns canned responses in sequence:

```python
def fake(system, user):
    # 1st call → seed preferences JSON
    # 2nd call → rerank response (with real job ids extracted from the prompt)
    # 3rd call → updated preferences after feedback
    # 4th call → second rerank response
```

The rerank response generator parses the actual user-prompt JSON to extract the real job IDs returned by BM25, so the test stays valid even if the underlying corpus or retrieval changes.

### What we deliberately don't test

- **The real LLM** — testing actual API responses would be flaky and expensive. Behaviour is tested through the *interface* (`chat_json` parsing + retry) and through *mocks* at the route boundary.
- **The frontend** — vanilla JS with no business logic worth automated testing; the brief explicitly says no FE polish.
- **Performance** — out of scope for a take-home; would matter for production.

### Why this matters for the interview

You can say in the call:

> *"The suite is 59 tests, runs in half a second, no API key needed. It covers the deterministic parts in depth — data loading, retrieval, filtering, JSON parsing, retry logic, and all five HTTP routes — and mocks the LLM at the route boundary so I can verify the full request lifecycle without burning API quota. That mocking layer also doubles as a contract test: if I change the rerank prompt or response shape, the API tests catch it."*

---

## 9. Known Limitations

1. **Sessions die on server restart.** In-memory only. Fine for demo, would need Redis for prod.
2. **No concurrency control.** If two requests for the same session arrive simultaneously, the second one overwrites the first. Demo is single-user so this doesn't matter.
3. **No retry on bad JSON.** If the LLM returns malformed JSON the request 500s. Production would need self-correction.
4. **BM25 has no negation.** `avoid` terms only influence the LLM rerank, not the BM25 retrieval stage. A job heavy in "enterprise" keywords can still surface in the top 30 if its other terms match well.
5. **No location normalisation.** "San Francisco" and "SF" and "Bay Area" are not recognised as the same place by the location filter.
6. **Description truncation at 600 chars in the rerank prompt** — sometimes useful context (responsibilities, tech stack) gets cut.
7. **Rate-limit retries are short.** One 5-second retry, then fail. On a constrained provider tier this fails sooner than necessary.
8. **No frontend tests.** Vanilla JS with no real business logic to test; intentional given the brief.
9. **LLM contract not enforced by test against real provider.** The mocked tests verify our parsing and routing, but if a provider silently changes its JSON output shape we'd only catch it at runtime.

---

## 10. Future Improvements

Roughly ordered by ROI:

### Easy wins (a few hours each)

1. **User-editable preferences UI** — let the user click chips in the right panel to add/remove `prefer`/`avoid` terms directly. Bypasses needing to write feedback for trivial changes.
2. **Manual hard filters** — checkboxes for "Remote only", "Sponsors visa", min salary slider. Doesn't need the LLM at all.
3. **Cache LLM responses** — same (candidate, prefs) → same rerank result. Saves cost during demo / development.
4. **Pagination / "see more"** — currently top 3, but the retriever returns 20. Show the next 7 on demand.
5. **Highlight matched terms** — bold the words in the job description that match the BM25 query so the user sees *why* it scored high.

### Medium effort (a day each)

6. **Hybrid retrieval (BM25 + embeddings)** — embed jobs once at startup, embed the query each round, combine scores `(0.5 * bm25 + 0.5 * cosine)`. Catches semantic matches BM25 misses. Best quality bump for the cost.
7. **Persist sessions to Redis** — one-line change to `sessions.py`, sessions survive restart, can scale horizontally.
8. **Eval harness** — set of (candidate, feedback, expected_attributes) tuples. Run nightly. Track precision-at-3 over time. Catches regressions in prompt changes.
9. **Self-correcting LLM** — if `chat_json` fails to parse, re-prompt with "your previous response was invalid, here it is, return valid JSON".
10. **Per-job "more like this"** — let user click a job and the system uses *that* job as a positive example for future ranking.

### Larger projects (multi-day)

11. **Switch to a stronger model for rerank** — GPT-4o or Claude Opus would give noticeably better rerank quality and reasoning. Trade-off: cost goes up 10×.
12. **Reranking with reranker models** — Cohere Rerank or BGE Reranker. Purpose-built for query↔doc reranking, often outperforms LLMs on this specific task at lower latency.
13. **Multi-objective ranking** — let the candidate weight different axes (skills vs comp vs stage). UI sliders update the rubric in the rerank prompt.
14. **Conversational refinement** — instead of just text feedback, full chat-style interaction. Could ask clarifying questions back ("you said less full-stack — does that mean no front-end at all, or just majority backend?").
15. **Learn-to-rank from implicit signals** — track which jobs users actually click through to. Use that as positive training data for a small XGBoost model that augments the LLM scoring.

### "If this were a real product"

16. **Auth + multi-tenant** — each user has their own session history, can come back, can save jobs.
17. **Job alerts** — given a saved profile, email weekly when matching jobs are added.
18. **Job ingestion pipeline** — scrape YC / Greenhouse / Lever continuously, dedupe, re-index.
19. **Telemetry + A/B testing** — track CTR per job, per ranker version, etc.
20. **Self-hosted LLM** — Llama 3 or Mistral on a local GPU for privacy + cost at scale.

---
