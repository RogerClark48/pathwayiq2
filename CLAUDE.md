# PathwayIQ — Claude Code Project Context

## What this project is

PathwayIQ is a career guidance tool built for **Scott Clark Consultants**. The first deployment is the **GM IoT Course and Career Explorer**: a web application that connects GM IoT course data with job market data (NCS and Prospects), helping prospective students understand the career pathways their courses lead to, and discover courses from a career starting point.

The app serves **GMIoT (Greater Manchester IoT)** course data (83 courses) alongside NCS and Prospects job data (1,252 records).

This is **v2**. V1 is preserved at `C:\Dev\pathwayiq`. V2 adds a structured qualification level layer and Skills England occupational progression data to ground course-career connections in real pathways, not just semantic similarity.

---

## Stack

| Component | Detail |
|---|---|
| Jobs database | SQLite — `job_roles_asset.db` — 1,252 records (1,216 with named content fields; 36 with NULL content) |
| Courses database | SQLite — `gmiot.sqlite` — 83 GMIoT courses |
| Connections database | SQLite — `connections.db` — pre-computed course→job connections (850 pairs, 83 courses) |
| SE data | SQLite — `se_data.db` — 951 SE occupations, 2,644 progression pairs, embeddings |
| Vector store | Chroma — `chroma_store/` — `gmiot_jobs` (~2,432 chunks) + `gmiot_courses` (166 chunks) |
| Embeddings | Voyage AI `voyage-3.5` (1024 dims) — cloud API (`VOYAGE_API_KEY` in `.env`) |
| LLM — chat & advisory | Claude Haiku (`claude-haiku-4-5-20251001`) — via Anthropic API |
| LLM — enrichment | Claude Sonnet (`claude-sonnet-4-6`) — advisory cards, data pipeline |
| Backend | Python (Flask) |
| Frontend | HTML/CSS/JS — mobile-first, vanilla JS |
| Dev environment | Windows, VS Code, Python venv at `C:\Dev\pathwayiq2\venv` |

**Ollama is retired.** All LLM inference is via the Anthropic API. No local models.

---

## API keys (`.env`)

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude Haiku/Sonnet — chat, advisory, and data pipeline |
| `VOYAGE_API_KEY` | Voyage AI — runtime query embedding |
| `SKILLS_ENGLAND_API_KEY` | Skills England Occupational Maps API — data pipeline only, no runtime calls |

---

## Anthropic API usage

**Chat and advisory:** Claude Haiku. Two-turn tool-use pattern — Haiku directs retrieval via `specify_searches`, backend executes, Haiku selects via `select_results`.

**Advisory cards:** Claude Sonnet. System-initiated, minimum 4-interaction threshold, 5-interaction spacing.

**Data pipeline:** Claude Haiku for bulk inference tasks (level tagging etc.). One API call per record. Never use CC inspection for bulk tasks — it exhausts context. Write a Python script; let Haiku do the inference.

```python
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=10,
    system="Your system prompt here",
    messages=[{"role": "user", "content": "Record content here"}]
)
result = response.content[0].text.strip()
```

---

## Voyage AI (query-time embedding)

All runtime embeddings use Voyage AI with `input_type="query"`. Chunks were embedded with `input_type="document"` during the pipeline.

```python
import voyageai
vo = voyageai.Client()  # reads VOYAGE_API_KEY from environment

result = vo.embed(["user query text"], model="voyage-3.5",
                  input_type="query", output_dimension=1024)
vector = result.embeddings[0]
```

---

## Data

### Jobs (`job_roles_asset.db`, table: `jobs`, PK: `id`)
- Two sources: **NCS** (National Careers Service) and **Prospects** — scraped from user-facing web pages, not structured API records
- The same job role can appear from both sources — intentional, do **not** deduplicate
- 1,252 rows total; 1,216 have named content fields; 36 have NULL content fields — handle gracefully
- No SOC codes, no ESCO codes — all data is prose derived from web pages

**Well-populated fields (1,216+ records):**

| Field | Notes |
|---|---|
| `id`, `title`, `normalized_title` | Core identity |
| `source`, `source_id`, `url` | NCS / Prospects provenance |
| `enriched_description` | Full enriched text blob |
| `overview`, `typical_duties`, `skills_required`, `entry_routes`, `salary`, `progression` | Named content fields — 100% populated for enriched records |

**Partially populated (~39–40%):**

| Field | Notes |
|---|---|
| `salary_min` / `salary_max` | 471 records — integers; 0 = null sentinel |
| `qualifications_summary` | 488 records |
| `typical_hours` | 416 records |

**v2 additions:**

| Field | Notes |
|---|---|
| `level` | RQF integer 1–7 — added in v2. 34 NULL (no content records). Assigned by Haiku from `entry_routes` + `qualifications_summary` + `progression`. |

**Everything else is empty or near-zero** (ESCO fields, work_environment, etc.).

- Salary: `salary_min`/`salary_max` integers; `salary` column = prose narrative
- Two Chroma chunks per job in `gmiot_jobs` collection:
  - `{id}_overview`: title + overview + typical_duties
  - `{id}_skills`: title + skills_required + entry_routes + progression

### Courses (`gmiot.sqlite`, table: `gmiot_courses`, PK: `course_id`)
- Source: GMIoT — 83 courses across 6 partner providers — scraped from user-facing web pages
- All named content fields 100% populated: `overview`, `what_you_will_learn`, `entry_requirements`, `progression`
- `level` field: 95% populated — 4 nulls are Short Courses where null is correct
- `mode`: 45% populated — delivery mode (full-time, part-time etc.)
- `esco_code`: empty throughout — not used
- SSA classification: `ssa_code` (1–15) + `ssa_label` (full text) — assigned by Sonnet
- Subject tile navigation matches on `ssa_label` (exact text match, no embedding)
- Qual tile navigation matches on `qual_type` IN list (see `QUAL_FILTER_MAP` in api.py)
- Two Chroma chunks per course in `gmiot_courses` collection:
  - `{course_id}_overview`: title + overview + what_you_will_learn
  - `{course_id}_skills`: title + entry_requirements + progression

### RQF level mapping (used throughout v2)

| Level | Equivalent |
|---|---|
| 1 | Entry level — no qualifications required |
| 2 | GCSE / Intermediate |
| 3 | A Level / T Level / Advanced |
| 4 | HNC |
| 5 | HND / Foundation Degree |
| 6 | Bachelor's degree |
| 7 | Master's / Postgraduate / Chartered |

### Cross-collection search
- Lift the stored vector by ID from one collection, query the other collection with it
- Validated — cross-collection scores typically 80–85%

### Skills England data (`se_data.db` — v2, Phase 1 complete)

Pulled from Skills England Occupational Maps API as a one-off pre-pass. No live runtime calls — app queries local SQLite only. Refresh is a periodic maintenance task.

**API:**
- Base URL: `https://occupational-maps-api.skillsengland.education.gov.uk/api/v1/`
- Auth header: `X-API-KEY` — value from `SKILLS_ENGLAND_API_KEY` in `.env`
- Swagger: `https://occupational-maps-api.skillsengland.education.gov.uk/swagger/index.html`

**Tables:**

`se_routes` — 15 SE route categories
| Column | Type | Notes |
|---|---|---|
| `route_id` | INTEGER PK | |
| `name` | TEXT | e.g. "Digital", "Engineering and manufacturing" |

`se_occupations` — 951 approved apprenticeship standards
| Column | Type | Notes |
|---|---|---|
| `std_code` | TEXT PK | e.g. OCC0534 — note some have letter suffixes (OCC0397A etc.) |
| `name` | TEXT | Occupation name |
| `level` | INTEGER | RQF level 2–7 (no Level 1 in SE standards). Distribution: L2=193, L3=316, L4=152, L5=55, L6=139, L7=96 |
| `route_id` | INTEGER FK | References `se_routes.route_id` |
| `typical_job_titles` | TEXT | Pipe-separated string e.g. "Financial accountant\|Tax adviser". NULL for ~200 records. |
| `embedding` | BLOB | Voyage AI `voyage-3.5` float32 (1024 dims) — name + typical_job_titles. All 951 rows populated. |

`se_progressions` — 2,644 directed pairs
| Column | Type | Notes |
|---|---|---|
| `std_code_from` | TEXT | Occupation that progresses FROM |
| `std_code_to` | TEXT | Occupation that progresses TO |

Compound PK on `(std_code_from, std_code_to)` — no duplicates.

**SSA → SE route mapping** (used in `tag_stdcodes.py`):
| SSA label | SE route |
|---|---|
| Engineering and Manufacturing Technologies | Engineering and manufacturing |
| Information and Communication Technology | Digital |
| Construction, Planning and the Built Environment | Construction and the built environment |
| Health, Public Services and Care | Health and science |
| Arts, Media and Publishing | Creative and design |
| Business, Administration and Law | None (use all occupations) |
| Preparation for Life and Work | None |
| Social Sciences | None |

**Pipeline scripts** (in `scripts/`):
- `pull_se_data.py` — pulls routes, occupations, progressions
- `assign_routes.py` — assigns route_id to occupations via Routes/{id}
- *(archived)* `pull_se_typical_titles.py`, `embed_se_occupations.py`, `embed_job_titles.py`, `tag_stdcodes.py` — moved to `scripts/archive/`; stdCode tagging approach retired

### `job_progression_cache` (`job_roles_asset.db`)

Demand-driven cache of Sonnet-generated progression results. One row per job. Populated on first user request per job — cache builds naturally through usage.

| Column | Type | Notes |
|---|---|---|
| `job_id` | INTEGER PK | FK to `jobs.id` |
| `narrative` | TEXT | 2–3 sentence plain-English guidance from Sonnet |
| `inbound_json` | TEXT | JSON array of `{"id": N, "title": "..."}` — roles that lead to this one |
| `outbound_json` | TEXT | JSON array of `{"id": N, "title": "..."}` — roles this leads to |
| `prompt_version` | INTEGER | 1 — increment when prompt changes to invalidate cache |
| `created_at` | TEXT | ISO timestamp |

Served by `GET /jobs/<id>/progression` in api.py. Candidates sourced from Chroma cross-collection search (top 30 nearest jobs). `prompt_version = 1` filter means old cache rows survive prompt changes without deletion — they just don't match the filter.

### `connections.db` — pre-computed course→job connections (v2)

| Column | Type | Notes |
|---|---|---|
| `course_id` | INTEGER | FK to gmiot_courses |
| `job_id` | INTEGER | FK to jobs |
| `semantic_score` | INTEGER | Domain similarity % (Chroma cross-collection) |
| `skills_score` | INTEGER | Skills alignment % (what_you_will_learn vs skills_required) |
| `created_at` | TEXT | Timestamp |

Compound PK on `(course_id, job_id)`. 850 pairs across 83 courses. Built by `scripts/build_connections.py` — semantic search + level gap filter (job.level ≤ course.level + 2) + Haiku domain gatekeeping. Served by `/courses/<id>/careers` in api.py with `"source": "connections_table"` in response; falls back to live Chroma search if no rows found.

---

## Chroma — known issue

Stale HNSW lock files can cause 500 errors on collection access. Reboot Windows before debugging code if you see `Nothing found on disk` errors from Chroma.

---

## Interface design (summary)

Full spec: `PathwayIQ_Interface_Design_Spec_v2.docx`

**Three zones:**
- Header (fixed) — brand mark + session overview icon
- Central area (scrollable) — all content: cards, detail views, advisory cards, chat bubbles
- Bottom bar (fixed) — one-line LLM response (italic teal) + text input bar

**Three levels of depth:**
1. Card headline — course/job name, provider/source, top 3 connections with match scores
2. Detail view — full enriched content from DB, triggered by tapping card title, expands in place
3. External link — deferred to bottom of detail view

**Card types:**
- Course card — title (tappable), qualification pill (amber), provider pill (grey), top 3 career connections with match score + salary + source label + source link
- Career card — title (tappable), salary (teal), source, top 3 course connections
- Advisory card — dashed amber border, system-initiated, no match score, amber CTA

**Navigation:**
- Vertical scroll thread, newest at bottom, earlier cards faded
- Transition labels between cards
- All external links open new tab

---

## Build sequence

### V1 — complete
| Phase | Scope |
|---|---|
| 1 | Python backend — query API endpoints |
| 2 | Minimal HTML/CSS front end — starting screen, course card, career card |
| 3 | Detail view — expand card in place |
| 4 | LLM integration — chat input, session context, bottom zone response line |
| 5 | Advisory cards |
| 6 | Session overview + save/pin |
| 7 | GMIoT data swap — gmiot.sqlite, named fields, Voyage AI, new Chroma collections |

### V2 — in progress
| Phase | Scope | Status |
|---|---|---|
| 0 | Qualification pathway map — modal overlay from qual grid trigger link | Complete |
| 1 | Skills England data layer — pull routes, occupations, progressions into `se_data.db` | Complete |
| 1b | Pre-computed connections table — course→job pairs with Haiku gatekeeping, served from connections.db | Complete |
| 1c | Progression card — demand-driven Sonnet generation, permanent cache in `job_progression_cache` | Complete |
| 2 | Improved card relevance — filter by occupation level and route | Not started |
| 3 | Progression advisory mode — Sonnet generates pathway narrative using SE data | Not started |
| 4 | Desktop/tablet progression map — visual network on wider screens | Not started |

---

## Runtime data flow

### Course card load — what happens end to end

1. User taps a subject tile or qual tile → frontend calls `GET /courses?subject=X` or `GET /courses?qual=Y`
2. `api.py` queries `gmiot.sqlite` → returns course list → frontend renders course cards
3. For each course card, frontend calls `GET /courses/<course_id>/careers?limit=3`
4. `api.py` checks `connections.db` for pre-computed connections:
   - **Hit:** returns top N jobs by `semantic_score`, marked `"source": "connections_table"`
   - **Miss:** falls back to live Chroma cross-collection search (lifts stored vector, queries `gmiot_jobs`, applies domain + skills floors)
5. Frontend renders career connections on the course card (job title, salary, source, match score)

### Chat flow

1. User submits a free-text query → `POST /chat`
2. Haiku (turn 1, tool use) — receives query + session context + browsing history → calls `specify_searches` tool to declare what to search for
3. `api.py` executes Chroma searches against `gmiot_courses` and/or `gmiot_jobs` collections using a Voyage AI query embedding
4. Haiku (turn 2, tool use) — receives candidates → calls `select_results` to pick the best matches and generate an acknowledgement
5. Response returned to frontend: selected results + one-line acknowledgement for the bottom bar

### Advisory card trigger

- After every qualifying interaction, session counter is checked
- Minimum 4 qualifying interactions before first advisory; minimum 5-interaction spacing thereafter
- Sonnet generates the advisory card content using recent session context

---

## api.py — path constants

All paths now point to v2. Current values in `api.py`:

```python
CHROMA_PATH    = r"C:\Dev\pathwayiq2\chroma_store"
GMIOT_DB       = r"C:\Dev\pathwayiq2\gmiot.sqlite"
JOBS_DB        = r"C:\Dev\pathwayiq2\job_roles_asset.db"
CONNECTIONS_DB = r"C:\Dev\pathwayiq2\connections.db"
```

`COURSES_DB` remains pointing to v1 `emiot.sqlite` — it is dead code in v2. The `course_row()` function that used it is unused; all course queries use `GMIOT_DB`. Do not remove `COURSES_DB` without also removing `course_row()`.

---

## Environment setup

### Start the server

```
C:\Dev\pathwayiq2\venv\Scripts\python.exe api.py
```

Server runs on `http://localhost:5000`. The frontend (`app/index.html`) is served statically by Flask — open it via the server, not directly from the filesystem, to avoid CORS issues.

### Activate the venv (for interactive use)

```
C:\Dev\pathwayiq2\venv\Scripts\activate
```

### Dependencies

All installed in `venv`. Key packages:

| Package | Purpose |
|---|---|
| `flask`, `flask-cors` | Web server |
| `chromadb` | Vector store client |
| `voyageai` | Embedding API client |
| `anthropic` | Claude API client |
| `numpy` | Cosine similarity in pipeline scripts |
| `python-dotenv` | `.env` loading |
| `requests`, `httpx` | HTTP — `requests` for pipeline scripts, `httpx` in api.py |
| `sqlite3` | Standard library — no install needed |

### Gotchas

- **Chroma lock files** — stale HNSW locks cause 500 errors. Reboot Windows if you see `Nothing found on disk` errors.
- **`.env` must be in project root** — `load_dotenv()` looks for it at `C:\Dev\pathwayiq2\.env`
- **venv is not committed** — recreate with `python -m venv venv` then `pip install flask flask-cors chromadb voyageai anthropic requests httpx python-dotenv`

---

## Key principles

- Mobile first
- Vertical scroll (not carousel)
- One result format regardless of input method (filter tap or free-text chat)
- Qualification names not level numbers in UI (T Level not Level 3, etc.)
- Subject area as entry point — not a marketing surface
- Advisory cards in central zone only — bottom zone is LLM response line + input bar only
- No live Skills England API calls at runtime — all SE data is pre-pulled and stored locally
- Bulk LLM inference tasks: always use explicit Haiku API calls via Python script, never CC inspection
