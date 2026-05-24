# Candidate Job Matcher

A take-home system that takes a candidate profile, returns the top 3 matching jobs,
accepts natural-language feedback, and iteratively refines its recommendations.

---

## Setup

```bash
# 1. Clone / unzip
cd candidate-job-matcher

# 2. Create virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your LLM key
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY  (or OPENAI_API_KEY + LLM_PROVIDER=openai)

# 5. Run
./run.sh
# or: uvicorn backend.app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## How to Use

1. **Pick a sample candidate** from the dropdown (Carin, Scott, or Warren), or
   expand **"Paste candidate JSON"** and drop in any raw LinkedIn-style profile JSON.
2. Click **Start Session**. The system calls the LLM once to seed an initial
   `Preferences` object from the profile.
3. Click **Get Recommendations**. The system runs BM25 retrieval → hard filters → LLM rerank
   and shows the top 3 jobs with reasons and concerns.
4. Type feedback in the box ("less enterprise", "remote only", "more backend AI")
   and click **Refine Recommendations**. Repeat as many times as you like.
5. The **right panel** always shows the current `Preferences` JSON (inspect how it evolves),
   the debug panel (BM25 query, how many jobs were retrieved/filtered), and the full
   feedback history with a before-vs-after diff.

---

## Run the eval script

```bash
python -m backend.eval
```

Runs a scripted 3-round loop (seed → initial recs → canned feedback → refined recs)
for each of the three sample candidates and prints titles, scores, and preference diffs.
Useful for sanity-checking that feedback meaningfully shifts the results.

---

## Run the tests

```bash
.venv/bin/pytest
# or
python -m pytest
```

59 tests, all offline (LLM is mocked), runs in under 1 second. Covers data loading,
BM25 retrieval, hard filters, LLM JSON parsing, retry logic, and all five HTTP routes.
See [DOCUMENTATION.md §8](DOCUMENTATION.md) for the full test-by-test breakdown.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/candidates` | List sample candidates |
| `POST` | `/session` | Create session. Body: `{"candidate_id": <int>}` or `{"candidate": {…}}` |
| `POST` | `/recommend` | Get top-3 jobs. Body: `{"session_id": "…"}` |
| `POST` | `/feedback` | Refine. Body: `{"session_id": "…", "feedback": "…"}` |
| `GET` | `/session/{id}` | Full session history for inspection |

### Swapping LLM provider

```env
# In .env:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Both providers use a thin `httpx` wrapper — no heavy SDKs. The prompt layer is
identical; only the API endpoint and auth header differ.

---

## Design Decisions

### Retrieval: BM25 (not embeddings)

The dataset is ~1 000 keyword-rich job postings. BM25 (`rank-bm25`) gives strong
recall with zero precompute, no API costs for retrieval, and sub-millisecond queries.

An embedding-based approach (semantic similarity) would require:
- an embedding model call per round (latency + cost)
- a pre-built vector index (complexity)
- little quality gain on this domain, since job titles and skill keywords are
  already highly discriminative terms

BM25 handles the heavy lifting; the LLM handles meaning.

### Ranking: LLM rerank over top-20

Rather than trying to make BM25 do the fine-grained ranking, we pass the top ~20
retrieved jobs to the LLM as a reranking task. The LLM:

- sees the full candidate profile + current `Preferences`
- scores each job 0–100 on four weighted axes:
  skill overlap (40%), seniority match (20%), company stage/type (20%), preference alignment (20%)
- returns `reasons[]` and `concerns[]` per job

This gives both quality ranking *and* transparent explanations in a single call.

### Structured preferences, not free-text vibes

After each feedback turn the LLM updates a typed `Preferences` object:

```json
{
  "must": { "remote_only": false, "require_sponsorship": false, "job_types": [], "excluded_locations": [] },
  "prefer": ["backend", "early-stage startup", "AI/ML"],
  "avoid": ["enterprise", "full-stack"],
  "free_text_notes": "Founding-engineer level roles at seed/series-A AI companies"
}
```

This object drives *both* the BM25 query (prefer-terms boost the query) *and*
the rerank prompt (prefer/avoid are explicit instructions). It also makes the
right-panel "Preferences" inspector immediately readable.

### Hard filters vs soft preferences

`must.*` fields (remote, sponsorship, job type, excluded locations) are applied as
hard post-BM25 filters before the LLM sees any jobs. This means the LLM never
wastes tokens on jobs that can't satisfy a hard constraint, and the filter logic
is deterministic and auditable.

Everything else (prefer, avoid, free_text_notes) is soft: it biases retrieval
and strongly influences reranking but doesn't hard-reject jobs.

### No DB, no embeddings, no auth

The brief explicitly says to skip these. All session state is in-memory Python dicts.
Restarting the server clears all sessions — entirely fine for local demo use.

---

## Tech stack

| Layer | Choice |
|-------|--------|
| API framework | FastAPI |
| ASGI server | Uvicorn |
| Retrieval | rank-bm25 |
| LLM client | httpx (raw REST, no SDK) |
| LLM provider | Gemini 2.0 Flash (default) / GPT-4o-mini |
| Frontend | Vanilla JS + Tailwind CDN |
| Data models | Pydantic v2 |
| Storage | In-memory dict |

---

## File structure

```
candidate-job-matcher/
  backend/
    app.py          # FastAPI routes
    models.py       # Pydantic types
    data.py         # Load jobs/candidates, build BM25 index
    retriever.py    # BM25 search + hard filters
    llm.py          # Gemini / OpenAI client
    prompts.py      # All LLM prompt functions
    sessions.py     # In-memory session store
    eval.py         # CLI eval script
  frontend/
    index.html      # Single-page UI
    app.js          # Vanilla JS
  data/
    jobs.json
    candidates.json
  requirements.txt
  .env.example
  run.sh
```
