# FutureFinder — Claude Code Project Context

**Active branch:** `visual-redesign` (current dev)
**Railway production:** `futurefinder-v3` branch

## What this project is

FutureFinder is a career guidance web application built for **Scott Clark Consultants**, deployed for **GMIoT (Greater Manchester Institute of Technology)**. It helps prospective students explore courses and understand where they lead careers-wise.

The AI persona is **Finn** — a conversational guide who interviews users to understand their interests, then surfaces relevant courses and career pathways.

---

## Stack

| Component | Detail |
|---|---|
| Courses DB | SQLite — `futurefinder.sqlite` — 82 active courses |
| Jobs DB | SQLite — `job_roles_asset.db` — 1,252 records |
| SE data | SQLite — `se_data.db` — 951 SE occupations, 2,644 progression pairs |
| Vector store | Chroma — `chroma_store/` — see collections below |
| Embeddings | Voyage AI `voyage-3.5` (1024 dims) — `VOYAGE_API_KEY` in `.env` |
| LLM — welcome interview | Claude Sonnet (`claude-sonnet-4-6`) |
| LLM — course selection | Claude Haiku (`claude-haiku-4-5-20251001`) |
| LLM — progression/advisory | Claude Sonnet |
| Backend | Python (Flask) — `api.py` |
| Frontend | Vanilla JS SPA — `app/` — mobile-first |
| Dev | Windows, VS Code, venv at `C:\Dev\pathwayiq2\venv` |

---

## API keys (`.env`)

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Sonnet + Haiku — all runtime LLM calls |
| `VOYAGE_API_KEY` | Voyage AI — runtime query embedding |
| `SKILLS_ENGLAND_API_KEY` | Data pipeline only — no runtime calls |

---

## Databases

### `futurefinder.sqlite` — courses (primary)

| Table | Rows | Notes |
|---|---|---|
| `courses` | 82 | Active GMIoT courses |
| `providers` | 6 | Partner institutions |
| `campuses` | 14 | Physical locations with lat/lng |
| `course_career_pathways` | 76 | Sonnet-generated career narratives + card jobs |
| `qual_vocab` | 20 | Canonical qualification type vocabulary |

**`courses` key fields:**
`course_id`, `course_title`, `provider_id`, `ssa_code` (int), `qual_type` (canonical), `level` (RQF int), `mode` (FT/PT/FT-PT), `overview`, `content`, `entry_requirements`, `progression`, `match_chunk` (Voyage-embedded text), `is_active`

SSA codes are the standard DfE taxonomy (1–15) plus custom 99 (Sustainability). Labels live in `SSA_LABELS` in `api.py` and mirrored in `app/ssa.js`.

### `job_roles_asset.db` — jobs

Table: `jobs`, PK: `id`. Two sources: NCS and Prospects (same role may appear from both — intentional, do not deduplicate). 1,252 rows; 1,216 with named content fields; 36 NULL.

Key fields: `id`, `title`, `source`, `level` (RQF), `overview`, `typical_duties`, `skills_required`, `entry_routes`, `salary`, `progression`, `salary_min`/`salary_max` (integers; 0 = null sentinel).

Also contains `ssa_categories` and `ssa_tier2` lookup tables used by `_build_active_subjects()`.

---

## Chroma collections

| Collection | Items | Used for |
|---|---|---|
| `match_courses` | 76 | **Primary** — one chunk per course, metadata: level/ssa_code/qual_type/mode. Used by pivot retrieval and WHERE-filtered search |
| `gmiot_jobs` | 2,432 | Job search — two chunks per job (`overview`, `skills`) |
| `gmiot_jobs_skills` | 1,216 | Skills chunks for job matching |
| `gmiot_jobs_duties` | 1,216 | Duties chunks |
| `gmiot_courses_learning` | 83 | LEGACY — only used by compute_skills_score |
| others | — | Archive/experimental — not used at runtime |

`match_courses` metadata fields: `title`, `provider`, `level` (int), `ssa_code` (int), `qual_type` (str), `mode` (str), `campus`. These are filterable via Chroma `where={}` clauses.

---

## `institution_config.py` — institution-specific settings

To deploy for a different institution, only this file changes:

| Setting | Current value | Purpose |
|---|---|---|
| `INSTITUTION_NAME` | `"GMIoT"` | Short name in prompts and UI |
| `INSTITUTION_FULL_NAME` | `"Greater Manchester Institute of Technology"` | Full name in welcome text and prompts |
| `INSTITUTION_REGION` | `"Greater Manchester"` | Region label in job detail view |
| `COURSES_DB` | `futurefinder.sqlite` | Path to courses database |
| `PROVIDERS` | 6 partner colleges | Provider name → location, used in prompts |
| `QUAL_FILTER_MAP` | qual tile labels → qual_type lists | Maps UI tile labels to DB qual_type values |

**Removed** (were here, now gone):
- `SSA_MAP` — dead code; current frontend uses chat not direct subject endpoint
- `SUBJECT_AREAS` — replaced by `_build_active_subjects()` which derives live from DB

Subject area labels are now driven by the DB (`_build_active_subjects()` at startup) and the standard `SSA_LABELS` dict in `api.py`. The `/api/welcome-data` endpoint returns an `institution` object `{full_name, abbrev, region}` so the frontend never hardcodes institution strings.

---

## api.py — path constants

```python
FUTUREFINDER_DB = r"C:\Dev\pathwayiq2\futurefinder.sqlite"   # courses (primary)
JOBS_DB         = r"C:\Dev\pathwayiq2\job_roles_asset.db"    # jobs
CHROMA_PATH     = r"C:\Dev\pathwayiq2\chroma_store"
ANALYTICS_DB    = r"C:\Dev\pathwayiq2\analytics.db"
CONNECTIONS_DB  = r"C:\Dev\pathwayiq2\connections.db"        # LEGACY
```

---

## Runtime data flow

### Welcome interview → course pivot

1. User opens app → `WelcomeView` loads, tiles built from `/api/welcome-data` (DB-driven SSA codes)
2. User types or taps a tile/chip → `POST /chat/welcome` with `session_id` + `message`
3. **Sonnet** runs the welcome interview (`_WELCOME_INTERVIEW_SYSTEM`): asks narrowing questions, detects interests
4. Sonnet emits one of three retrieval markers:
   - `[PIVOT_TO_COURSES]` — semantic search via Voyage AI + Chroma `match_courses`
   - `[FILTER:N]` — SQL: all active courses for SSA code N
   - `[SHOW_QUAL_MAP]` — show qualification pathway map
5. Optional filter markers accumulate in session: `[LEVEL:N]`, `[MODE:PT/FT]`, `[QUAL:X]`, `[X:ALL]` to clear
6. On pivot: **Haiku** selects best 5–8 from Chroma candidates using full conversation context
7. Frontend renders course cards in `StartChatView`; user can tap through to detail, career view, job detail

### Session filter system

Active filters `{level, mode, qual}` persist in the welcome session dict across turns. They apply as:
- **Chroma** `where=` clause in `retrieve_courses_for_pivot()` via `build_chroma_where()`
- **SQL** `WHERE` clauses in `get_filtered_courses()`

Level filter uses a ±1 range via `level_range(n)` — `[LEVEL:4]` queries levels 3–5.
Active filters are injected into Finn's `dynamic_note` each turn so he knows what's constraining results and can surface it to the user.

### Career / job detail

- Course detail → `GET /courses/<id>/detail` → returns full course + career pathways from `course_career_pathways`
- Job detail → `GET /jobs/<id>` → returns job content + related GMIoT courses (Chroma cross-collection)
- Career view → `GET /courses/<id>/careers` → course→job connections from `course_career_pathways.card_jobs`

---

## LLM usage

### Sonnet — welcome interview (`welcome_chat_llm`)
Runs `_WELCOME_INTERVIEW_SYSTEM` prompt. Persona: Finn. Detects user intent across up to 4 turns, emits retrieval markers, accumulates session filters. Uses prompt caching (ephemeral) on the base system prompt.

### Haiku — course selection (`retrieve_courses_for_pivot`)
Given 25 Chroma candidates + full conversation, selects best 5–8 via `select_courses` tool. Also used for advisory cards and progression generation.

### Sonnet — progression / advisory
`/jobs/<id>/progression` — generates career ladder. Advisory cards system-initiated after qualifying interactions.

### Data pipeline
Haiku for bulk inference (level tagging, course enrichment etc.). Always write a Python script — never use CC inspection for bulk tasks.

```python
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=10,
    system="...",
    messages=[{"role": "user", "content": "..."}]
)
```

---

## Frontend — `app/`

Vanilla JS SPA with hash-based routing (`router.js`). Key views:

| View | File | Purpose |
|---|---|---|
| `WelcomeView` | `views/welcome.js` | Landing — Finn greeting, subject tiles, starter chips |
| `StartChatView` | `views/start-chat.js` | Chat thread — Finn bubbles, course card pivot |
| `CourseCarouselView` | `views/course-carousel.js` | Course map + swipe cards |
| `CourseDetailView` | `views/course-detail.js` | Full course detail |
| `CareerView` | `views/career-view.js` | Career pathways from a course |
| `JobDetailView` | `views/job-detail.js` | Job profile + related courses |
| `PathwayMapView` | `views/pathway-map.js` | Qualification level map |
| `SavedListView` | `views/saved-list.js` | Saved courses and careers |

Key modules: `api.js` (fetch helpers, memoised `loadWelcomeData`), `state.js`, `subjects.js` (SSA colour/icon metadata), `ssa.js` (SSA_LABELS), `dom.js` (`renderField`, `renderProse`, `splitProse`).

**Institution strings** are never hardcoded in the frontend. All views read `getWelcomeData()?.institution` which returns `{full_name, abbrev, region}` from the API.

---

## RQF level reference

| Level | Qualifications |
|---|---|
| 2 | GCSE / Intermediate |
| 3 | T Level / A Level — entry point at GMIoT |
| 4 | HNC / Higher Apprenticeship |
| 5 | HND / Foundation Degree |
| 6 | Bachelor's Degree / Degree Apprenticeship |
| 7 | Master's / Postgraduate |

GMIoT starts at Level 3 — no GCSEs or A Levels offered.

---

## Environment setup

```
# Start server
C:\Dev\pathwayiq2\venv\Scripts\python.exe api.py
# Serves on http://localhost:5000 — open via server not filesystem (CORS)

# Activate venv
C:\Dev\pathwayiq2\venv\Scripts\activate
```

Key packages: `flask`, `flask-cors`, `chromadb`, `voyageai`, `anthropic`, `python-dotenv`, `requests`, `httpx`

**Gotchas:**
- Chroma stale HNSW locks → `Nothing found on disk` errors → reboot Windows
- `.env` must be in project root
- `analytics.db` resets on Railway deploy — needs a Volume to persist

---

## Key principles

- Mobile-first, chat-first
- Finn is the persona — warm, on-task, never invents course details
- Subject areas derived from live DB at startup — never hardcoded
- Institution strings always from `institution_config.py` via API — never hardcoded in frontend
- Suggestion chips must be first-person sentences expressing user intent (5–8 words)
- Tile seeds: `"I'd like to explore ${label} courses"` — not bare label strings
- No live Skills England API calls at runtime — SE data pre-pulled to `se_data.db`
- Bulk LLM inference: always write a Python script using Haiku directly, never CC inspection
- `connections.db` is LEGACY — course→job connections now served from `course_career_pathways` in `futurefinder.sqlite`
