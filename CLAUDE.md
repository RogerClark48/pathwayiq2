# PathwayIQ — Claude Code Project Context

## What this project is

PathwayIQ is a career guidance tool built for **Scott Clark Consultants**. The first deployment is the **GM IoT Course and Career Explorer**: a web application that connects GM IoT course data with job market data (NCS and Prospects), helping prospective students understand the career pathways their courses lead to, and discover courses from a career starting point.

The app serves **GMIoT (Greater Manchester IoT)** course data (83 courses) alongside NCS and Prospects job data (1,252 records).

This is **v2**. V1 is preserved at `C:\Dev\pathwayiq`. V2 adds a structured qualification level layer and Skills England occupational progression data to ground course-career connections in real pathways, not just semantic similarity.

---

## Stack

| Component | Detail |
|---|---|
| Jobs database | SQLite — `emiot_jobs_asset.db` — 1,252 records (1,216 with named content fields; 36 with NULL content) |
| Courses database | SQLite — `gmiot.sqlite` — 83 GMIoT courses |
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

### Jobs (`emiot_jobs_asset.db`, table: `jobs`, PK: `id`)
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

### Skills England data (v2 — to be built in Phase 1)
- Pulled from Skills England Occupational Maps API as a one-off pre-pass — no live runtime calls
- Stored locally in SQLite
- Tables: `se_occupations` (stdCode, name, level, route), `se_progressions` (std_code_from, std_code_to)
- API base: `https://occupational-maps.skillsengland.education.gov.uk/api/v1/`
- Key endpoint: `GET /OccupationsBySOC?SOCCode={code}&expand=occupation.typicaljobtitles,occupation.ssa`

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
| 0 | Qualification pathway map — modal overlay from qual grid trigger link | Brief written, ready for CC |
| 1 | Skills England data layer — pull routes, occupations, progressions into SQLite | Not started |
| 2 | Improved card relevance — filter by occupation level and route | Not started |
| 3 | Progression advisory mode — Sonnet generates pathway narrative using SE data | Not started |
| 4 | Desktop/tablet progression map — visual network on wider screens | Not started |

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
