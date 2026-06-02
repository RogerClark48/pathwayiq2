import json
import os
import re
import sqlite3
import time
from datetime import datetime
import httpx
import numpy as np
import voyageai
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, session, redirect, url_for, make_response
from flask_cors import CORS
from threading import Lock
import chromadb

load_dotenv()
print(f"[startup] VOYAGE_API_KEY present: {bool(os.environ.get('VOYAGE_API_KEY'))}", flush=True)
print(f"[startup] All env vars: {[k for k in os.environ.keys()]}", flush=True)

from institution_config import (
    INSTITUTION_NAME, INSTITUTION_FULL_NAME, INSTITUTION_REGION,
    PROVIDERS, SSA_MAP, SUBJECT_AREAS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BASE              = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH        = os.path.join(_BASE, "chroma_store")
JOBS_DB            = os.path.join(_BASE, "job_roles_asset.db")
CONNECTIONS_DB     = os.path.join(_BASE, "connections.db")  # LEGACY — superseded by course_career_pathways in futurefinder.sqlite
LMI_DB             = os.path.join(_BASE, "lmi.db")
FUTUREFINDER_DB    = os.path.join(_BASE, "futurefinder.sqlite")
ANALYTICS_DB       = os.path.join(_BASE, "analytics.db")
VOYAGE_MODEL       = "voyage-3.5"
VOYAGE_DIMS        = 1024
MIN_SCORE                = 50   # recalibrated for Voyage AI voyage-3.5 (was 65 for nomic-embed-text)
COURSE_CHAT_MIN_SCORE    = 45   # recalibrated for Voyage AI voyage-3.5 (was 55)
TOP_N_CANDIDATES         = 8
SEARCH_TOP_N             = 8  # results per specified search (Stage 2+)

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL      = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL        = "claude-haiku-4-5-20251001"
SONNET_MODEL       = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Shared base — applies to every LLM call in the welcome flow.
# Change tone, persona, or institution context here once; all calls pick it up.
# ---------------------------------------------------------------------------
# Build subject areas text from actual courses in DB — stays correct when
# the course set changes or the app is deployed for a different institution.
def _build_active_subjects() -> str:
    """
    For each Tier 1 area present in active courses:
    - Single Tier 2 sub-category present → use the Tier 2 label (more specific).
    - Multiple Tier 2 sub-categories present → use the Tier 1 label (reflects breadth).
    """
    try:
        ff   = sqlite3.connect(FUTUREFINDER_DB)
        jobs = sqlite3.connect(JOBS_DB)

        tier1_codes = [r[0] for r in ff.execute(
            "SELECT DISTINCT ssa_code FROM courses WHERE is_active=1 AND ssa_code IS NOT NULL ORDER BY ssa_code"
        ).fetchall()]

        lines = []
        for t1 in tier1_codes:
            tier2_codes = [r[0] for r in ff.execute(
                "SELECT DISTINCT ssa_tier2_code FROM courses "
                "WHERE is_active=1 AND ssa_code=? AND ssa_tier2_code IS NOT NULL",
                (t1,),
            ).fetchall()]

            if len(tier2_codes) == 1:
                row = jobs.execute(
                    "SELECT label FROM ssa_tier2 WHERE tier2_code=?", (tier2_codes[0],)
                ).fetchone()
            else:
                row = jobs.execute(
                    "SELECT label FROM ssa_categories WHERE ssa_code=?", (t1,)
                ).fetchone()

            if row:
                lines.append(f"- {row[0]}")

        ff.close()
        jobs.close()
        return "\n".join(lines) if lines else "- Engineering, Digital, Construction, Health, Arts, Business"
    except Exception:
        return "- Engineering, Digital, Construction, Health, Arts, Business"

_ACTIVE_SUBJECTS = _build_active_subjects()

# ---------------------------------------------------------------------------
_FF_BASE_SYSTEM = """\
You are FutureFinder, an AI assistant helping prospective students explore
courses and careers at the Greater Manchester Institute of Technology (GMIoT).

You are talking to someone who may be:

- A school leaver finishing A Levels or T Levels
- An adult learner considering returning to study
- Someone changing careers
- Someone uncertain about what they want

GMIoT offers courses from Level 3 (T Level) through Level 7 (Master's), including
apprenticeships at higher levels. There are no GCSEs or A Levels on offer.

GMIoT's subject areas are:
""" + _ACTIVE_SUBJECTS + """

When exploring what a user wants, stay strictly within these subject areas.

- If a user expresses an interest GMIoT cannot serve (e.g. agriculture, land
  management, catering, travel and tourism), be honest: tell them GMIoT does
  not offer courses in that area. Do not stretch to a "nearest equivalent".
  Ask about other interests or experience that might relate to one of the subjects
  GMIoT does offer. If after a further turn the user still cannot express a relevant
  interest, offer to show the subject areas so they can see what is available.
- When suggesting alternatives or asking narrowing questions, only name areas
  from the list above. Never suggest subject areas not on this list (e.g. do
  not suggest languages, catering, hospitality, law as standalone areas).
- Do not generate sub-area suggestion chips within a subject unless you are
  certain GMIoT has courses there. When in doubt, pivot to courses and let the
  results speak — do not invent categories.

Courses are retrieved from a database — do not invent course names or details
from your own knowledge. Only work with courses the system provides to you.

## Saved items

The user may have saved courses or careers during this session. If so, they are
listed in the context as "Saved items". Treat them as confirmed interest.

- Use them as a reliable prior when the user's input is vague or ambiguous.
- Do not re-recommend courses the user has already saved.
- If saved items cluster around a subject or level, treat that as the user's
  revealed preference — use it to break ties in selection, not to narrate back.
- Only surface them explicitly when it genuinely adds something, e.g. if the
  user says "I'm not sure where to go next", it is appropriate to say something
  like "You've been looking at a few engineering courses — want me to suggest
  what could come after those?"

Do not use emojis.
Do not promise outcomes (e.g. "this will lead to a job in…").\
"""

_WELCOME_INTERVIEW_SYSTEM = _FF_BASE_SYSTEM + """

## Your goal

Map the user's interest onto the GMIoT subject areas above, then trigger
the appropriate response. Any signal is usable — a subject, a job title, a
work-style preference, a constraint. Pivot as soon as you have enough to act
on. Do not hold out for richer input.

## How to narrow

When a user gives a broad signal, ask one question to narrow it. Frame
questions around **work style and context**, not invented sub-categories:

Good narrowing angles:
- Hands-on practical work vs design/planning vs desk-based analysis
- Working with people vs working with systems/technology vs working
  with physical materials
- Indoors vs outdoors
- Building/making things vs maintaining/fixing things vs managing people

Do NOT generate sub-category lists from your world knowledge (e.g. do not
say "are you thinking mechanical, electrical, civil, or software?"). Those
sub-categories may not match what GMIoT offers and will mislead the user.
Let the retrieval system surface the actual courses once you have a broad
direction.

## Tone and length

Warm but on-task. 2–3 sentences per response. The user is on a phone.
Do not probe personal circumstances. Do not ask how they're doing.
Respect autonomy — if they say "I don't know" twice, offer to browse.

## Escalation

**Turn 1:** If usable input, pivot immediately. If vague, ask one
narrowing question with a concrete example or two.

**Turn 2:** If still vague, try a different angle — negative elicitation
often works: "Anything you'd rather not do? Sit at a desk, work outdoors
in all weathers, deal with the public?"

**Turn 3:** Offer to browse: "Want me to show you what GMIoT has across
all its subject areas? You can see what's there and go from there."

**Turn 4:** Suggest an advisor: "Sometimes it's easier to talk this
through with someone. GMIoT advisors can help you figure out where to
start — [book a free course chat](https://gmiot.ac.uk/book-your-course-chat/)."

Abandon the escalation the moment the user gives you something to work with.

## Safeguarding

If the user discloses sensitive content (mental health, family difficulties,
anything beyond course advice), acknowledge briefly with warmth, redirect to
the course/career frame, and offer the advisor link if appropriate. Do not
probe, validate at length, or engage as a counsellor.

If the user asks about student support or who to speak to:
[book a free course chat](https://gmiot.ac.uk/book-your-course-chat/)

## Triggering course results — three markers

**[PIVOT_TO_COURSES]** — Use when you have a specific interest or need to
find the best-matching courses from across the full catalogue. The retrieval
system will run a semantic search and present the most relevant results.
Use this for: specific topics, roles, work-style driven searches, any search
that is not a request to browse a whole subject area.

Example: "Got it — you want hands-on engineering work. Let me find the
courses that fit best. [PIVOT_TO_COURSES]"

**[FILTER:N]** — Use only when the user explicitly asks to browse the
complete list for a subject area — phrases like "show me all engineering
courses", "what do you have in health", "show me everything in construction". This returns the
full unranked list for that area. Do not use it simply because the user
named a subject area — if they named it as a preference or in response to
a narrowing question, use [PIVOT_TO_COURSES] instead so the retrieval can
use the full conversation context. This can return a long list — use
sparingly, and only if the user has not responded to a narrowing question.

Subject area codes:
- Health, Public Services and Care → [FILTER:1]
- Engineering and Manufacturing Technologies → [FILTER:4]
- Construction, Planning and the Built Environment → [FILTER:5]
- Information and Communication Technology → [FILTER:6]
- Sport, Leisure and Recreation → [FILTER:8]
- Arts, Media and Publishing → [FILTER:9]
- Social Sciences → [FILTER:11]
- Business, Administration and Law → [FILTER:15]
- Sustainability → [FILTER:99]

For anything more specific than a whole subject area, use [PIVOT_TO_COURSES]
instead — e.g. "software development courses" → [PIVOT_TO_COURSES], not
[FILTER:6].

Do not use [FILTER:N] and [PIVOT_TO_COURSES] in the same response.

**[SHOW_QUAL_MAP]** — Use when the user asks about qualification types,
levels, or how qualifications relate to each other. Do not combine with
[PIVOT_TO_COURSES] or [FILTER:N].

## Suggestion chips

Use [SUGGESTIONS:option|option|option] (2–4 options) when offering the user
concrete things to choose between. Keep each option short (3–5 words).

Chips must come from one of two sources only:
1. **GMIoT's subject areas** — use the names from the list above.
2. **Work-style dimensions** — e.g. hands-on vs desk-based, people-facing
   vs technical, indoors vs outdoors.

Do not use [SUGGESTIONS:...] with [PIVOT_TO_COURSES] or [FILTER:N].

## What not to do

- Do not ask the user's name, age, or location.

## Post-pivot advisory mode

Once courses have been shown (signalled in the dynamic note), the interview
is over. Answer whatever the user asks helpfully — preparation, entry
requirements, what a career involves, anything. You may still use
[FILTER:N] or [PIVOT_TO_COURSES] if the user wants to see more courses.

Use the advisor booking link only for genuinely institution-specific
questions you cannot answer (deadlines, specific entry exceptions, bursaries).
Not as a general deflection.

Keep responses concise — 3–4 sentences. Mobile-readable.
"""

_SELECT_COURSES_TOOL = {
    "name": "select_courses",
    "description": "Select courses from the candidate set that best match what the user said in the conversation. Read the conversation first, then choose.",
    "input_schema": {
        "type": "object",
        "properties": {
            "selected_course_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Course IDs in order of relevance (most relevant first). "
                    "Select 3 to 8 courses. If fewer than 5 genuinely match the student's subject, return only the ones that do — do not pad with unrelated courses."
                ),
            },
            "intro_text": {
                "type": "string",
                "description": (
                    "One sentence introducing the list to the user. "
                    "Example: 'Based on your interest in engineering, here are some courses to explore.' "
                    "Plain English, warm, short, mobile-readable."
                ),
            },
        },
        "required": ["selected_course_ids", "intro_text"],
    },
}

PROGRESSION_SYSTEM_PROMPT = (
    "You are a career guidance advisor helping college students understand career pathways. "
    "You give warm, honest, plain-English advice grounded in how careers actually develop. "
    "The job profiles you receive include two authoritative fields written by career experts at "
    "the National Careers Service and Prospects: 'Entry routes' describes how people actually get "
    "into this role, and 'Career progression' describes where this role leads. "
    "These fields are your primary source for progression — use them to shape both your "
    "selection of inbound/outbound roles and the language of your narrative. Draw on the specific "
    "routes, qualifications, and next steps they describe. Where your own knowledge adds useful "
    "context or more current detail — such as emerging roles, updated qualification routes, or "
    "recent industry trends — you may supplement the expert content, but do not contradict it. "
    "You must respond with valid JSON only. "
    "Do not use markdown code blocks, backticks, or any text outside the JSON object itself."
)

# SSA_MAP imported from institution_config

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
app.secret_key   = os.environ.get("SECRET_KEY", "dev-fallback-key")
ADMIN_PASSWORD   = os.environ.get("ADMIN_PASSWORD", "admin")
AUTH_ENABLED     = os.environ.get("AUTH_ENABLED", "false").lower() == "true"

vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
jobs_col               = chroma.get_collection("gmiot_jobs")
jobs_skills_col        = chroma.get_collection("gmiot_jobs_skills")
match_courses_col      = chroma.get_collection("match_courses")
courses_learning_col   = chroma.get_collection("gmiot_courses_learning")  # LEGACY — only used by compute_skills_score via /courses/<id>/careers

# Ensure job_progression_cache table exists (was dropped in v3 cleanup)
_jpc_conn = sqlite3.connect(JOBS_DB)
_jpc_conn.execute("""
    CREATE TABLE IF NOT EXISTS job_progression_cache (
        job_id         INTEGER PRIMARY KEY,
        narrative      TEXT,
        inbound_json   TEXT,
        outbound_json  TEXT,
        prompt_version INTEGER,
        created_at     TEXT,
        explain_text   TEXT
    )
""")
for _col in ("courses_json TEXT", "explain_cache_version INTEGER", "courses_cache_version INTEGER"):
    try:
        _jpc_conn.execute(f"ALTER TABLE job_progression_cache ADD COLUMN {_col}")
    except Exception:
        pass  # column already exists
_jpc_conn.commit()
_jpc_conn.close()

EXPLAIN_CACHE_VERSION  = 1  # bump when the explain prompt changes to force regeneration
COURSES_CACHE_VERSION  = 2  # bump when the job_courses Haiku prompt changes

CAUTION_DIVERGENCE_THRESHOLD  = 15  # domain% - skills% > this → caution flag
CROSS_COLLECTION_MIN_SKILLS   = 72  # hard floor — connections below this are excluded
CROSS_COLLECTION_MIN_DOMAIN   = 75  # hard floor — low domain score connections excluded

# ---------------------------------------------------------------------------
# Anthropic API wrapper — rate-limit detection and usage logging
# ---------------------------------------------------------------------------

# Cost per million tokens (USD) — update when Anthropic changes pricing
_MODEL_PRICING = {
    "claude-sonnet-4-6": {
        "input": 3.00, "output": 15.00,
        "cache_write": 3.75, "cache_read": 0.30,
    },
    "claude-haiku-4-5-20251001": {
        "input": 0.80, "output": 4.00,
        "cache_write": 1.00, "cache_read": 0.10,
    },
}

class RateLimitError(Exception):
    pass


def _anthropic_post(payload: dict, call_site: str, session_id: str | None = None, timeout: float = 30.0):
    """POST to Anthropic API. Raises RateLimitError on 429; raises for other HTTP errors."""
    resp = httpx.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after", "?")
        print(f"[RATE_LIMIT] call_site={call_site} retry_after={retry_after}s session={str(session_id or '')[:8]}", flush=True)
        try:
            conn = sqlite3.connect(ANALYTICS_DB)
            conn.execute(
                "INSERT INTO events (session_id, ts, event, meta) VALUES (?, ?, ?, ?)",
                (
                    session_id or "system",
                    datetime.utcnow().isoformat(),
                    "rate_limit",
                    json.dumps({"call_site": call_site, "retry_after": retry_after}),
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        raise RateLimitError(call_site)
    resp.raise_for_status()
    try:
        model  = payload.get("model", "")
        usage  = resp.json().get("usage", {})
        inp    = usage.get("input_tokens", 0)
        out    = usage.get("output_tokens", 0)
        c_write = usage.get("cache_creation_input_tokens", 0)
        c_read  = usage.get("cache_read_input_tokens", 0)
        rates  = _MODEL_PRICING.get(model, {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0})
        cost   = (
            inp     * rates["input"]       / 1_000_000 +
            out     * rates["output"]      / 1_000_000 +
            c_write * rates["cache_write"] / 1_000_000 +
            c_read  * rates["cache_read"]  / 1_000_000
        )
        conn   = sqlite3.connect(ANALYTICS_DB)
        conn.execute(
            "INSERT INTO api_usage (ts, session_id, call_site, model, input_tokens, output_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), session_id or "system", call_site, model, inp + c_write + c_read, out, cost),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
    return resp


# ---------------------------------------------------------------------------
# Per-user session store
# ---------------------------------------------------------------------------
_sessions      = {}
_sessions_lock = Lock()
SESSION_TTL    = 1800  # 30 minutes inactivity


def get_session(session_id: str) -> dict:
    """Get or create isolated session state for this session_id."""
    with _sessions_lock:
        now = time.time()
        if session_id not in _sessions:
            _sessions[session_id] = {
                "qualifying_count":            0,
                "advisory_count":              0,
                "interactions_since_last":     0,
                "seen_ids":                    [],
                "last_context":                [],
                "last_active":                 now,
            }
        else:
            _sessions[session_id]["last_active"] = now
        return _sessions[session_id]


def cleanup_sessions() -> None:
    """Remove sessions inactive for more than SESSION_TTL seconds."""
    with _sessions_lock:
        now     = time.time()
        expired = [sid for sid, s in _sessions.items()
                   if now - s["last_active"] > SESSION_TTL]
        for sid in expired:
            del _sessions[sid]
        if expired:
            print(f"[session] cleaned up {len(expired)} expired sessions. "
                  f"Active: {len(_sessions)}", flush=True)


# ---------------------------------------------------------------------------
# Welcome interview session store
# ---------------------------------------------------------------------------
_welcome_sessions      = {}
_welcome_sessions_lock = Lock()


def get_welcome_session(session_id: str) -> dict:
    with _welcome_sessions_lock:
        now = time.time()
        if session_id not in _welcome_sessions:
            _welcome_sessions[session_id] = {
                "messages":           [],
                "interview_turn_count": 0,
                "pivot_done":         False,
                "created_at":         now,
                "last_used_at":       now,
            }
        else:
            _welcome_sessions[session_id]["last_used_at"] = now
        return _welcome_sessions[session_id]


def cleanup_welcome_sessions() -> None:
    with _welcome_sessions_lock:
        now     = time.time()
        expired = [sid for sid, s in _welcome_sessions.items()
                   if now - s["last_used_at"] > SESSION_TTL]
        for sid in expired:
            del _welcome_sessions[sid]
        if expired:
            print(f"[welcome_session] cleaned up {len(expired)} expired. "
                  f"Active: {len(_welcome_sessions)}", flush=True)


def welcome_chat_llm(session_id: str, message: str, saved_items: list | None = None) -> dict:
    """
    Append user message to the welcome session, call Sonnet, strip the
    [PIVOT_TO_COURSES] marker, increment turn count, persist bot reply.
    Returns {"bot_response": str, "pivot_to_courses": bool}.
    """
    sess = get_welcome_session(session_id)

    with _welcome_sessions_lock:
        sess["messages"].append({"role": "user", "content": message})
        history = list(sess["messages"])

    saved_items = saved_items or []
    saved_note  = ""
    if saved_items:
        titles = ", ".join(i["title"] for i in saved_items if i.get("title"))
        if titles:
            saved_note = f"\n\nSaved items: {titles}"

    print(f"[welcome_chat] session={session_id[:8]}... turn={sess['interview_turn_count']+1} "
          f"msg={message!r} saved={len(saved_items)}", flush=True)

    if sess.get("pivot_done"):
        dynamic_note = "\n\n[Courses have been shown. You are now in advisory mode — see ## Post-pivot advisory mode in your instructions. You may still use [FILTER:N] and [PIVOT_TO_COURSES] markers if the user asks to see courses.]" + saved_note
    else:
        dynamic_note = (
            f"\n\n[This is interview turn {sess['interview_turn_count'] + 1}. "
            f"At turn 4 or beyond with no usable input, use the graceful exit.]"
            + saved_note
        )
    try:
        resp = _anthropic_post({
            "model":    SONNET_MODEL,
            "system": [
                {
                    "type": "text",
                    "text": _WELCOME_INTERVIEW_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": dynamic_note,
                },
            ],
            "messages":   history,
            "max_tokens": 200,
            "temperature": 0.5,
        }, call_site="welcome_chat", session_id=session_id)
        raw_text = resp.json()["content"][0]["text"].strip()
    except RateLimitError:
        return {"bot_response": "The service is very busy right now — please try again in a moment.", "pivot_to_courses": False}
    except Exception as e:
        print(f"[welcome_chat] Sonnet error: {e}", flush=True)
        return {"bot_response": None, "pivot_to_courses": False}

    filter_match      = re.search(r'\[FILTER:(\d+)\]', raw_text)
    suggestions_match = re.search(r'\[SUGGESTIONS:([^\]]+)\]', raw_text)
    filter_code   = int(filter_match.group(1)) if filter_match else None
    suggestions   = [s.strip() for s in suggestions_match.group(1).split('|') if s.strip()] if suggestions_match else []
    pivot         = "[PIVOT_TO_COURSES]" in raw_text
    show_qual_map = "[SHOW_QUAL_MAP]" in raw_text
    bot_response  = re.sub(r'\[FILTER:\d+\]', '', raw_text)
    bot_response  = re.sub(r'\[SUGGESTIONS:[^\]]+\]', '', bot_response)
    bot_response  = bot_response.replace("[PIVOT_TO_COURSES]", "").replace("[SHOW_QUAL_MAP]", "").strip()

    with _welcome_sessions_lock:
        sess["messages"].append({"role": "assistant", "content": bot_response})
        sess["interview_turn_count"] += 1

    print(f"[welcome_chat] pivot={pivot} filter_code={filter_code} suggestions={suggestions} show_qual_map={show_qual_map} response={bot_response[:80]!r}", flush=True)
    return {"bot_response": bot_response, "pivot_to_courses": pivot, "filter_code": filter_code, "suggestions": suggestions, "show_qual_map": show_qual_map}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def embed(text: str) -> list[float]:
    print(f"[embed] Voyage query embed: {text[:80]!r}", flush=True)
    try:
        result = vo.embed(
            [text],
            model=VOYAGE_MODEL,
            input_type="query",
            output_dimension=VOYAGE_DIMS,
        )
        vec = result.embeddings[0]
        print(f"[embed] OK — vector len={len(vec)}", flush=True)
        return vec
    except Exception as e:
        print(f"[embed] FAILED — {e}", flush=True)
        raise


def score(distance: float) -> int:
    return round((1 - distance) * 100)


def get_stored_vector(collection, chunk_id: str) -> list | None:
    """Lift a stored embedding vector by chunk ID. Returns None if not found."""
    result = collection.get(ids=[chunk_id], include=["embeddings"])
    if result["embeddings"] is not None and len(result["embeddings"]) > 0:
        return result["embeddings"][0]
    return None


def _cosine_similarity(vec_a, vec_b) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


# LEGACY — called only by /courses/<id>/careers, which is not used by the chat-first frontend.
# Course→job connections are now served from course_career_pathways in futurefinder.sqlite.
def compute_skills_score(course_id, job_id) -> int | None:
    """
    Skills alignment score: what_you_will_learn (course) vs skills_required (job).
    Returns integer percentage or None if vectors unavailable.
    """
    learning_vec = get_stored_vector(courses_learning_col, f"{course_id}_learning")
    skills_vec   = get_stored_vector(jobs_skills_col,      f"{job_id}_skills_only")
    if learning_vec is None or skills_vec is None:
        return None
    return round(_cosine_similarity(learning_vec, skills_vec) * 100)


def salary_string(low, high, currency="GBP") -> str | None:
    symbol = "£" if currency in ("GBP", "") else currency + " "
    low  = float(low  or 0)
    high = float(high or 0)
    if low == 0 and high == 0:
        return None
    if low == 0:
        return f"Up to {symbol}{int(high):,}"
    if high == 0:
        return f"From {symbol}{int(low):,}"
    return f"{symbol}{int(low):,} – {symbol}{int(high):,}"


def ff_course_row(course_id: str) -> dict | None:
    conn = sqlite3.connect(FUTUREFINDER_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT c.*, p.provider_name "
        "FROM courses c "
        "LEFT JOIN providers p ON c.provider_id = p.provider_id "
        "WHERE c.course_id = ? AND c.is_active = 1", (course_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def job_row(job_id: str) -> dict | None:
    conn = sqlite3.connect(JOBS_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, title, source, url, salary_min, salary_max, salary_currency, "
        "overview, typical_duties, skills_required, entry_routes, salary, progression "
        "FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def lmi_employer_text(job_id: int) -> str | None:
    try:
        conn = sqlite3.connect(LMI_DB)
        row = conn.execute(
            "SELECT employer_text FROM job_employer_text WHERE job_id = ?", (job_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def format_course_from_db(db: dict, match_score: int) -> dict:
    """Build a course result dict from a futurefinder.sqlite courses row."""
    return {
        "type":               "course",
        "id":                 db["course_id"],
        "title":              db["course_title"],
        "provider":           db.get("provider_name") or "",
        "level":              db.get("level"),
        "qualification_type": db.get("qual_type"),
        "source_url":         db.get("course_url"),
        "match_score":        match_score,
        "overview":           (db.get("overview") or "")[:500],
    }


def keyword_course_search(q: str, qualification: str | None) -> list[dict]:
    """SQLite LIKE search on course_title in courses. Returns exact-title matches first."""
    conn = sqlite3.connect(FUTUREFINDER_DB)
    conn.row_factory = sqlite3.Row
    sql = ("SELECT c.*, p.provider_name FROM courses c "
           "LEFT JOIN providers p ON c.provider_id = p.provider_id "
           "WHERE c.course_title LIKE ? AND c.is_active = 1")
    params: list = [f"%{q}%"]
    if qualification:
        sql += " AND c.qual_type = ?"
        params.append(qualification)
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        return []

    q_lower = q.lower()
    results = []
    for row in rows:
        db    = dict(row)
        exact = db["course_title"].lower() == q_lower
        results.append(format_course_from_db(db, match_score=100 if exact else 95))
    results.sort(key=lambda r: 0 if r["match_score"] == 100 else 1)
    return results


def format_course(meta: dict, db: dict | None, match_score: int) -> dict:
    result = {
        "type":               "course",
        "id":                 meta["course_id"],
        "title":              meta["course_name"],
        "provider":           meta["provider"],
        "subject_area":       meta.get("subject_area"),
        "level":              meta.get("level"),
        "qualification_type": meta.get("qualification_type"),
        "ssa_category":       meta.get("ssa_category"),
        "source_url":         meta.get("url"),
        "match_score":        match_score,
    }
    if db:
        result["overview"] = (db.get("overview") or "")[:500]
    return result


def format_job(meta: dict, db: dict | None, match_score: int) -> dict:
    sal = None
    if db:
        sal = salary_string(db.get("salary_min"), db.get("salary_max"),
                            db.get("salary_currency", "GBP"))
    result = {
        "type":        "job",
        "id":          meta["job_id"],
        "title":       meta["title"],
        "source":      meta.get("source", "").upper(),
        "source_url":  meta.get("url"),
        "match_score": match_score,
    }
    if sal:
        result["salary"] = sal
    if db:
        result["overview"] = (db.get("summary") or db.get("description") or "")[:400]
    return result




def format_browsing_history(browsing_history: list) -> str:
    """Format browsing history as readable text for the Haiku system prompt."""
    if not browsing_history:
        return "None yet."
    return "\n".join(
        f"  {item.get('type', 'item').capitalize()}: {item.get('title', '')}"
        for item in browsing_history
    )



_CHAT_TOOL = {
    "name": "submit_chat_result",
    "description": (
        "Submit the gatekeeping decision and acknowledgement after analysing "
        "the query, session context, and retrieved candidates."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "approved_job_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of job candidates approved as subject-relevant and level-appropriate.",
            },
            "approved_course_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of course candidates approved as subject-relevant and level-appropriate.",
            },
            "rejected_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of candidates rejected as subject-irrelevant or level-mismatched.",
            },
            "acknowledgement": {
                "type": "string",
                "description": "One-line confirmation for the bottom zone. What was understood, or what was found.",
            },
            "is_off_topic": {
                "type": "boolean",
                "description": "True if the query has no relevance to courses or careers. No search results will be shown.",
            },
        },
        "required": [
            "approved_job_ids",
            "approved_course_ids",
            "rejected_ids",
            "acknowledgement",
            "is_off_topic",
        ],
    },
}


_providers_text = "\n".join(
    f"{name} — {location}" for name, location in PROVIDERS.items()
)
_subjects_text = "\n".join(
    f"{label} — {desc}" for label, desc in SUBJECT_AREAS
)

_EXPLAIN_SYSTEM = (
    f"You are a course and career guidance advisor for {INSTITUTION_FULL_NAME}. "
    "A student has asked a question about qualifications, "
    "career pathways, or how the education system works.\n\n"
    "Answer clearly and warmly in 2–4 sentences. If it is natural to do so, end with "
    "a short suggestion of what the user could explore next — but do not force it.\n\n"

    "UK QUALIFICATION LEVELS (RQF):\n"
    "Level 1 — Entry level, no prior qualifications needed\n"
    "Level 2 — GCSE / Intermediate\n"
    "Level 3 — A Level, T Level, Advanced — typical university entry point\n"
    "Level 4 — HNC, Higher Apprenticeship\n"
    "Level 5 — HND, Foundation Degree, Higher Apprenticeship\n"
    "Level 6 — Bachelor's Degree, Degree Apprenticeship\n"
    "Level 7 — Master's Degree, Postgraduate, Chartered\n\n"

    "QUALIFICATION TYPES IN THIS APP:\n"
    "T Level — 2-year Level 3 vocational qualification, equivalent to 3 A levels; "
    "includes a 45-day industry placement. Strong technical grounding.\n"
    "Apprenticeship — Work-based learning: the student is employed and studies "
    "alongside work. Available at Levels 2 through 7.\n"
    "HNC (Higher National Certificate) — Level 4; typically 1 year full-time or "
    "2 years part-time. Often a stepping stone to HND or degree top-up.\n"
    "HND (Higher National Diploma) — Level 5; typically 2 years full-time. "
    "Can top up to a full bachelor's degree in 1 additional year.\n"
    "HTQ (Higher Technical Qualification) — employer-designed Level 4–5 qualifications; "
    "HNCs and HNDs can carry HTQ status, signalling strong employer endorsement.\n"
    "Foundation Degree (FdA / FdSc) — Level 5; typically 2 years. "
    "Designed with employers; can top up to a bachelor's in 1 year.\n"
    "CertHE / DipHE — Level 4 / Level 5 certificates and diplomas of higher education.\n"
    "Access to HE Diploma — Level 3; designed for adults (typically 19+) returning to "
    "education after a break. Primary pathway into university for mature students.\n"
    "Bachelor's Degree (BA Hons, BEng Hons, BSc Hons) — Level 6; typically 3 years.\n"
    "Master's Degree (MSc) — Level 7; typically 1 year full-time postgraduate study.\n"
    "Short Course / Award — Short professional or skills-based courses, no fixed level.\n\n"

    f"PARTNER PROVIDERS (all in {INSTITUTION_REGION}):\n"
    f"{_providers_text}\n\n"

    "SUBJECT AREAS COVERED:\n"
    f"{_subjects_text}\n\n"

    "JOB DATA SOURCES:\n"
    "NCS — National Careers Service; UK government careers information\n"
    "Prospects — UK graduate careers website with detailed job role information\n\n"

    "Do not invent course titles, job titles, or facts not grounded in the above. "
    "If you genuinely do not know, say so briefly and suggest the user explore the app. "
    "Do not use markdown formatting — no bold, no bullet points, plain text only."
)


def chat_explain(message: str, chat_history: list, max_tokens: int = 300) -> str:
    """Direct Haiku call to answer a qualifications/pathway question. No tool use, no search.

    Returns plain text answer, or a fallback string on failure.
    """
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]
    messages = [{"role": m["role"], "content": m["content"]} for m in chat_history]
    messages.append({"role": "user", "content": message})
    try:
        resp = _anthropic_post({
            "model":      HAIKU_MODEL,
            "max_tokens": max_tokens,
            "system":     _EXPLAIN_SYSTEM,
            "messages":   messages,
        }, call_site="chat_explain")
        return resp.json()["content"][0]["text"].strip()
    except RateLimitError:
        return "The service is very busy right now — please try again in a moment."
    except Exception as e:
        print(f"[chat_explain] FAILED — {e}", flush=True)
        return "I'm not able to answer that right now — try exploring the subject areas or qualifications above."






_ADVISORY_TOOL = {
    "name": "submit_advisory_decision",
    "description": (
        "Submit one advisory item to proactively surface to the user — "
        "a course or career they haven't seen yet that genuinely adds value to their "
        "exploration. Only submit if there is a clear, high-quality match. "
        "If no candidate is worth surfacing, submit advisory_item_type='none'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "advisory_trigger": {
                "type": "string",
                "description": "Brief internal note on the session pattern that triggered this advisory (not shown to user).",
            },
            "advisory_item_type": {
                "type": "string",
                "enum": ["course", "job", "none"],
                "description": "'course', 'job', or 'none' if no strong match found.",
            },
            "advisory_item_id": {
                "type": "string",
                "description": "Exact ID of the course or job to surface. Empty string if advisory_item_type is 'none'.",
            },
            "advisory_explanation": {
                "type": "string",
                "description": "One sentence (max 20 words) shown to the user explaining why this is relevant to their exploration.",
            },
        },
        "required": [
            "advisory_trigger",
            "advisory_item_type",
            "advisory_item_id",
            "advisory_explanation",
        ],
    },
}


def build_advisory_candidates(session_context: list, seen_ids: list) -> list:
    """Retrieve candidates from both collections based on session context.

    Concatenates last 5 context titles as query, retrieves from both collections,
    excludes already-seen advisory IDs, returns top 10 by score.
    """
    if not session_context:
        return []

    context_query = " ".join(str(x) for x in session_context[-5:])
    try:
        vector = embed(context_query)
    except Exception:
        return []

    seen_set   = {str(x) for x in seen_ids}
    candidates = []

    try:
        job_hits = jobs_col.query(
            query_embeddings=[vector],
            n_results=10,
            where={"chunk": {"$eq": "overview"}},
            include=["metadatas", "distances", "documents"],
        )
        for id_, meta, dist, doc in zip(
            job_hits["ids"][0], job_hits["metadatas"][0],
            job_hits["distances"][0], job_hits["documents"][0],
        ):
            jid = str(meta["job_id"])
            if jid in seen_set:
                continue
            candidates.append({
                "type":      "job",
                "id":        jid,
                "title":     meta.get("title", ""),
                "score":     score(dist),
                "full_text": doc[:400],
            })
    except Exception as e:
        print(f"[advisory] job Chroma query failed: {e}", flush=True)

    try:
        course_hits = match_courses_col.query(
            query_embeddings=[vector],
            n_results=10,
            include=["metadatas", "distances", "documents"],
        )
        for cid, meta, dist, doc in zip(
            course_hits["ids"][0], course_hits["metadatas"][0],
            course_hits["distances"][0], course_hits["documents"][0],
        ):
            if cid in seen_set:
                continue
            candidates.append({
                "type":      "course",
                "id":        cid,
                "title":     meta.get("title", ""),
                "score":     score(dist),
                "full_text": doc[:300],
            })
    except Exception as e:
        print(f"[advisory] course Chroma query failed: {e}", flush=True)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:10]


def advisory_llm_call(session_context: list, candidates: list) -> dict | None:
    """Sonnet call to select one advisory item from candidates.

    Returns dict with type, id, explanation — or None if no good match / failure.
    """
    if not candidates:
        return None

    lines = []
    for c in candidates:
        if c["type"] == "job":
            lines.append(
                f"TYPE=job JOB_ID={c['id']} | {c['title']} ({c['score']}%)\n"
                f"{c['full_text']}"
            )
        else:
            lines.append(
                f"TYPE=course COURSE_ID={c['id']} | {c['title']} "
                f"({c.get('qualification_type', '')}, Level {c.get('level', '')}) ({c['score']}%)\n"
                f"{c['full_text']}"
            )

    context_block = "\n".join(f"  - {x}" for x in session_context[-5:])

    system_prompt = (
        "You are a proactive career guidance advisor. A user has been exploring courses "
        "and careers. Your job is to identify one item they haven't seen yet that "
        "genuinely adds value to their exploration — an unexpected but relevant connection.\n\n"
        "Only surface something if it's a high-quality, non-obvious match that opens a "
        "new angle on what they've been exploring. If nothing stands out clearly, "
        "submit advisory_item_type='none'.\n\n"
        "Rules:\n"
        "- Select at most one item — the single best addition to their exploration\n"
        "- Prefer items that open a new angle (adjacent job family, progression route, "
        "complementary qualification)\n"
        "- advisory_explanation: one sentence, max 20 words, specific to what they've "
        "been exploring\n"
        "- Never invent facts about courses or careers"
    )

    user_prompt = (
        f"What the user has explored this session:\n{context_block}\n\n"
        "Advisory candidates (not yet seen by this user):\n\n"
        + "\n\n---\n\n".join(lines)
    )

    try:
        resp = _anthropic_post({
            "model":       SONNET_MODEL,
            "max_tokens":  300,
            "system":      system_prompt,
            "tools":       [_ADVISORY_TOOL],
            "tool_choice": {"type": "tool", "name": "submit_advisory_decision"},
            "messages":    [{"role": "user", "content": user_prompt}],
        }, call_site="advisory_llm")
        result      = resp.json()["content"][0]["input"]
        item_type   = result.get("advisory_item_type", "none")
        item_id     = (result.get("advisory_item_id") or "").strip()
        explanation = result.get("advisory_explanation", "")
        trigger     = result.get("advisory_trigger", "")
        print(f"[advisory_llm] trigger={trigger!r} type={item_type} id={item_id!r}", flush=True)

        if item_type == "none" or not item_id:
            return None
        return {"type": item_type, "id": item_id, "explanation": explanation}
    except RateLimitError:
        return None
    except Exception as e:
        print(f"[advisory_llm] Sonnet call failed ({e}) — skipping", flush=True)
        return None


def _increment_qualifying(session_id: str) -> None:
    """Increment per-user qualifying counters."""
    sess = get_session(session_id)
    with _sessions_lock:
        sess["qualifying_count"] += 1
        if sess["advisory_count"] > 0:
            sess["interactions_since_last"] += 1
        print(
            f"[advisory] qualifying_count={sess['qualifying_count']} "
            f"advisory_count={sess['advisory_count']} "
            f"since_last={sess['interactions_since_last']}",
            flush=True,
        )


def check_advisory(session_context: list, session_id: str) -> dict | None:
    """Check if an advisory card should be triggered this interaction.

    Reads and writes per-user session state. Returns enriched advisory dict or None.
    Minimum 4 qualifying interactions before first advisory;
    minimum 5 between subsequent advisories.
    """
    sess = get_session(session_id)

    with _sessions_lock:
        if session_context:
            sess["last_context"] = list(session_context[-10:])

        qualifying_count = sess["qualifying_count"]
        seen_ids         = list(sess["seen_ids"])
        advisory_count   = sess["advisory_count"]
        since_last       = sess["interactions_since_last"]
        ctx              = session_context or list(sess["last_context"])

    if qualifying_count < 4:
        print(f"[advisory] skip — qualifying_count={qualifying_count} < 4", flush=True)
        return None

    if advisory_count > 0 and since_last < 5:
        print(f"[advisory] skip — interactions_since_last={since_last} < 5", flush=True)
        return None

    if not ctx:
        print("[advisory] skip — ctx empty (no session_context and no last_context)", flush=True)
        return None

    print(
        f"[advisory] checking — qualifying={qualifying_count} advisory={advisory_count} "
        f"since_last={since_last} ctx_len={len(ctx)}",
        flush=True,
    )
    candidates = build_advisory_candidates(ctx, seen_ids)
    if not candidates:
        print("[advisory] no candidates after exclusions", flush=True)
        return None

    advisory = advisory_llm_call(ctx, candidates)

    # Reset gap counter regardless of outcome — prevents hammering Sonnet
    with _sessions_lock:
        sess["interactions_since_last"] = 0
        sess["advisory_count"]         += 1

    if not advisory:
        return None

    with _sessions_lock:
        sess["seen_ids"] = seen_ids + [advisory["id"]]

    # Enrich with DB data
    if advisory["type"] == "job":
        db = job_row(advisory["id"])
        if not db:
            return None
        advisory["title"]      = db["title"]
        advisory["source"]     = (db.get("source") or "").upper()
        advisory["source_url"] = db.get("url")
        sal = salary_string(db.get("salary_min"), db.get("salary_max"),
                            db.get("salary_currency", "GBP"))
        if sal:
            advisory["salary"] = sal
    else:
        db = ff_course_row(advisory["id"])
        if not db:
            return None
        advisory["title"]              = db["course_title"]
        advisory["provider"]           = db.get("provider_name") or ""
        advisory["qualification_type"] = db.get("qual_type")
        advisory["source_url"]         = db.get("course_url")

    return advisory


def chat_llm_call(message: str, candidates: list,
                  chat_history: list | None = None,
                  browsing_history: list | None = None,
                  saved_items: dict | None = None) -> tuple:
    """Single Anthropic API call combining intent parsing and result gatekeeping.

    Uses tool use to guarantee a structured response — no JSON parsing needed.
    candidates:       list of dicts with keys type, id, title, score, full_text,
                      and (for courses) qualification_type, level
    chat_history:     list of {role, content} dicts — prior turns; trimmed to
                      last 10 exchanges (20 messages) before sending
    browsing_history: list of {type, title, id} dicts, oldest first
    saved_items:      {courses: [{title, id}], careers: [{title, id}]}
    Returns: (approved_job_ids, approved_course_ids, acknowledgement, is_off_topic)
    """
    chat_history     = chat_history or []
    browsing_history = browsing_history or []
    saved_items      = saved_items or {"courses": [], "careers": []}
    # Trim to last 10 exchanges (20 messages) to stay within token limits
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]
    lines = []
    for c in candidates:
        if c["type"] == "job":
            lines.append(
                f"TYPE=job JOB_ID={c['id']} | {c['title']} ({c['score']}%)\n"
                f"{c['full_text'][:500]}"
            )
        else:
            lines.append(
                f"TYPE=course COURSE_ID={c['id']} | {c['title']} "
                f"({c.get('qualification_type', '')}, Level {c.get('level', '')}) ({c['score']}%)\n"
                f"{c['full_text'][:300]}"
            )

    saved_courses_str = (
        ", ".join(c["title"] for c in saved_items.get("courses", []))
        or "none"
    )
    saved_careers_str = (
        ", ".join(c["title"] for c in saved_items.get("careers", []))
        or "none"
    )
    context_block = (
        "\n\nSESSION CONTEXT\n\n"
        f"Browsing history (oldest to newest):\n{format_browsing_history(browsing_history)}\n\n"
        "Saved items — the user explicitly saved these, indicating stronger interest:\n"
        f"  Courses: {saved_courses_str}\n"
        f"  Careers: {saved_careers_str}\n\n"
        "Use this to interpret follow-up queries — infer the subject domain and user "
        "intent from browsing and saved patterns when the message is short or ambiguous."
    )

    system_prompt = (
        "You are a career and course guidance assistant. A user has sent a message "
        "and a retrieval system has fetched candidate courses and careers.\n\n"
        "Perform three tasks in one pass:\n"
        "1. INTENT — determine what type of results the user wants:\n"
        "   - 'courses': user asks about studying, courses, qualifications, or training\n"
        "   - 'jobs': user asks about careers, jobs, roles, work, or salary\n"
        "   - 'both': general subject interest with no clear courses/jobs preference\n"
        "2. SUBJECT GATEKEEPING — remove candidates whose subject domain is clearly "
        "unrelated to the user's query and session context\n"
        "3. LEVEL GATEKEEPING — only apply when the user has explicitly stated their "
        "background in their message or session context (e.g. qualifications held: "
        "'I've just finished my A-levels', 'I have a degree in...'; experience: "
        "'I'm currently working as...', 'I'm a graduate...'; or explicit level preference: "
        "'I'm looking for entry-level', 'postgraduate options'). "
        "When no background is stated, approve candidates on subject relevance alone — "
        "do not infer or assume a user's level. Absence of information is not a signal.\n"
        "When background IS stated, apply level filtering directionally: "
        "approve candidates at or above the user's level (a user finishing an HNC should "
        "see degree-level progression routes — aspirational results are a feature). "
        "Only reject candidates that are clearly a backward step — significantly below "
        "the user's stated level. Never filter upward.\n\n"
        "TYPE ROUTING:\n"
        "- Intent is 'courses' → set approved_job_ids to [] (courses only)\n"
        "- Intent is 'jobs'    → set approved_course_ids to [] (jobs only)\n"
        "- Intent is 'both'    → approve relevant candidates from both types\n\n"
        "QUALIFICATION LEVEL REFERENCE:\n"
        "T Level → Level 3 | Higher Apprenticeship → Level 4–5 | "
        "HNC → Level 4 | HND → Level 5 | Bachelor's Degree → Level 6 | Master's Degree → Level 7\n\n"
        "Rules:\n"
        "- IDs must be exact numbers from the candidate list — never invent or alter IDs\n"
        "- acknowledgement: one short natural sentence (max 15 words) shown in the bottom bar\n"
        "- is_off_topic: true only if the message has nothing to do with courses, careers, "
        "or education\n"
        "- When in doubt, approve — only reject clear mismatches\n"
        "- Never invent facts about specific courses or careers"
    )

    user_prompt = (
        f"User message: \"{message}\"{context_block}\n\n"
        "Retrieved candidates:\n\n"
        + "\n\n---\n\n".join(lines)
    )

    print(f"[chat_llm] user_prompt=\n{user_prompt}", flush=True)

    try:
        resp = _anthropic_post({
            "model":       HAIKU_MODEL,
            "max_tokens":  1000,
            "system":      system_prompt,
            "tools":       [_CHAT_TOOL],
            "tool_choice": {"type": "tool", "name": "submit_chat_result"},
            "messages":    chat_history + [{"role": "user", "content": user_prompt}],
        }, call_site="chat_llm")
        result           = resp.json()["content"][0]["input"]
        approved_jobs    = [str(x) for x in result.get("approved_job_ids", [])]
        approved_courses = [str(x) for x in result.get("approved_course_ids", [])]
        ack              = result.get("acknowledgement", "")
        is_off_topic     = bool(result.get("is_off_topic", False))
        print(f"[chat_llm] tool call received — approved_jobs={approved_jobs} approved_courses={approved_courses}", flush=True)
        return approved_jobs, approved_courses, ack, is_off_topic
    except RateLimitError:
        return [], [], "The service is very busy right now — please try again in a moment.", False
    except Exception as e:
        print(f"[chat_llm] API call failed ({e}) — approving all", flush=True)
        approved_jobs    = [c["id"] for c in candidates if c["type"] == "job"]
        approved_courses = [c["id"] for c in candidates if c["type"] == "course"]
        return approved_jobs, approved_courses, "Here are some results for you.", False


# ---------------------------------------------------------------------------
# Access code authentication
# ---------------------------------------------------------------------------

_AUTH_EXEMPT = {"/access", "/logout"}

@app.before_request
def require_auth():
    """Gate all routes behind access code authentication."""
    if not AUTH_ENABLED:
        return
    if request.path in _AUTH_EXEMPT:
        return
    if request.path.startswith("/admin/"):
        return  # admin routes handle their own auth
    if session.get("authenticated"):
        return
    return redirect(url_for("access_page"))


@app.route("/access", methods=["GET", "POST"])
def access_page():
    error = None
    if request.method == "POST":
        submitted = (request.form.get("code") or "").strip().lower()
        conn = sqlite3.connect(ANALYTICS_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM access_codes WHERE LOWER(code) = ?", (submitted,)
        ).fetchone()
        if not row:
            error = "Code not recognised"
        elif row["expires_at"] and datetime.utcnow().isoformat() > row["expires_at"]:
            error = "This code has expired"
        else:
            conn.execute(
                "UPDATE access_codes SET used_count = used_count + 1 WHERE code = ?",
                (row["code"],)
            )
            conn.commit()
            conn.close()
            session["authenticated"] = True
            return redirect(url_for("serve_index"))
        conn.close()

    return make_response(_render_access_page(error))


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("access_page"))


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

import base64
import random

_WORDLIST = [
    "maple","cedar","amber","river","stone","cloud","bloom","spark","frost","heron",
    "solar","dunes","tidal","grove","ember","delta","crest","flint","haven","lunar",
]

def _check_admin_auth():
    """Returns True if the request carries valid Basic Auth for admin."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        _, password = decoded.split(":", 1)
        return password == ADMIN_PASSWORD
    except Exception:
        return False


@app.route("/admin/codes", methods=["GET", "POST"])
def admin_codes():
    if not _check_admin_auth():
        resp = make_response("Unauthorised", 401)
        resp.headers["WWW-Authenticate"] = 'Basic realm="PathwayIQ Admin"'
        return resp

    new_code = None
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row

    if request.method == "POST":
        action = request.form.get("action", "create")
        if action == "delete":
            code_to_delete = (request.form.get("code") or "").strip()
            if code_to_delete:
                conn.execute("DELETE FROM access_codes WHERE code = ?", (code_to_delete,))
                conn.commit()
        else:
            label    = (request.form.get("label") or "").strip() or "Unlabelled"
            days_str = (request.form.get("days") or "").strip()
            expires_at = None
            if days_str:
                try:
                    from datetime import timedelta
                    expires_at = (datetime.utcnow() + timedelta(days=int(days_str))).isoformat()
                except ValueError:
                    pass
            word     = random.choice(_WORDLIST)
            digits   = str(random.randint(1000, 9999))
            new_code = f"{word}-{digits}"
            now      = datetime.utcnow().isoformat()
            conn.execute(
                "INSERT INTO access_codes (code, label, expires_at, created_at, used_count) "
                "VALUES (?, ?, ?, ?, 0)",
                (new_code, label, expires_at, now)
            )
            conn.commit()

    rows = conn.execute(
        "SELECT code, label, expires_at, created_at, used_count FROM access_codes "
        "ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    now_iso = datetime.utcnow().isoformat()
    table_rows = ""
    for r in rows:
        if r["expires_at"]:
            expires_display = r["expires_at"][:16].replace("T", " ")
            status = "Expired" if now_iso > r["expires_at"] else "Active"
            status_cls = "expired" if status == "Expired" else "active"
        else:
            expires_display = "Permanent"
            status = "Active"
            status_cls = "active"
        highlight = ' style="background:#fffbe6;font-weight:bold;"' if r["code"] == new_code else ""
        code_val = r['code']
        delete_btn = (
            f'<form method="POST" action="/admin/codes" style="display:inline" '
            f'onsubmit="return confirm(\'Revoke {code_val}?\')">'
            f'<input type="hidden" name="action" value="delete">'
            f'<input type="hidden" name="code" value="{code_val}">'
            f'<button type="submit" class="btn-revoke">Revoke</button>'
            f'</form>'
        )
        table_rows += (
            f"<tr{highlight}>"
            f"<td>{r['code']}</td>"
            f"<td>{r['label']}</td>"
            f"<td>{expires_display}</td>"
            f"<td>{r['used_count']}</td>"
            f"<td class='{status_cls}'>{status}</td>"
            f"<td>{delete_btn}</td>"
            f"</tr>\n"
        )

    new_code_html = ""
    if new_code:
        new_code_html = (
            f'<p style="margin:12px 0;padding:10px 14px;background:#d1fae5;'
            f'border:1px solid #6ee7b7;border-radius:6px;font-family:monospace;font-size:16px;">'
            f'New code: <strong>{new_code}</strong></p>'
        )

    html = _ADMIN_PAGE_HTML.replace("{{TABLE_ROWS}}", table_rows).replace("{{NEW_CODE}}", new_code_html)
    return make_response(html)


@app.get("/admin/analytics")
def admin_analytics():
    if not _check_admin_auth():
        resp = make_response("Unauthorised", 401)
        resp.headers["WWW-Authenticate"] = 'Basic realm="FutureFinder Admin"'
        return resp

    days_param = request.args.get("days", "30")
    if days_param == "all":
        cutoff = None
        period_label = "All time"
    else:
        try:
            n = int(days_param)
        except ValueError:
            n = 30
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(days=n)).isoformat()
        period_label = f"Last {n} days"

    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row

    def q(sql, params=()):
        return conn.execute(sql, params).fetchall()

    ts_filter  = "ts >= ?" if cutoff else "1=1"
    ts_args    = (cutoff,) if cutoff else ()

    # Summary stats
    summary = conn.execute(
        f"SELECT COUNT(DISTINCT session_id) as sessions, COUNT(*) as total_events "
        f"FROM events WHERE {ts_filter}", ts_args
    ).fetchone()

    chat_submits = conn.execute(
        f"SELECT COUNT(*) as n FROM events WHERE event='chat_submit' AND {ts_filter}", ts_args
    ).fetchone()["n"]

    rate_limit_count = conn.execute(
        f"SELECT COUNT(*) as n FROM events WHERE event='rate_limit' AND {ts_filter}", ts_args
    ).fetchone()["n"]

    # Daily sessions — last 14 days regardless of period
    from datetime import timedelta
    cutoff14 = (datetime.utcnow() - timedelta(days=13)).strftime("%Y-%m-%d")
    daily_rows = q(
        "SELECT substr(ts,1,10) as day, COUNT(DISTINCT session_id) as n "
        "FROM events WHERE event='session_start' AND substr(ts,1,10) >= ? "
        "GROUP BY day ORDER BY day",
        (cutoff14,)
    )
    daily_max = max((r["n"] for r in daily_rows), default=1)
    daily_chart = ""
    for r in daily_rows:
        pct = int(r["n"] / daily_max * 100)
        daily_chart += (
            f'<div class="bar-row">'
            f'<span class="bar-label">{r["day"][5:]}</span>'
            f'<div class="bar-wrap"><div class="bar" style="width:{pct}%"></div>'
            f'<span class="bar-val">{r["n"]}</span></div></div>\n'
        )
    if not daily_chart:
        daily_chart = '<p class="empty">No session data yet.</p>'

    # Engagement funnel
    funnel_events = [
        ("session_start",      "Sessions started"),
        ("chat_submit",        "Chat messages sent"),
        ("course_impression",  "Courses shown"),
        ("career_impression",  "Careers shown"),
        ("course_detail_open", "Course details opened"),
        ("career_detail_open", "Career details opened"),
        ("progression_open",   "Progression views"),
        ("adzuna_click",       "Job listing clicks"),
    ]
    counts_by_event = {
        r["event"]: r["n"]
        for r in q(f"SELECT event, COUNT(*) as n FROM events WHERE {ts_filter} GROUP BY event", ts_args)
    }
    funnel_rows = ""
    for ev, label in funnel_events:
        n = counts_by_event.get(ev, 0)
        funnel_rows += f"<tr><td>{label}</td><td class='num'>{n}</td></tr>\n"

    # Top careers
    career_rows = q(
        f"SELECT entity_title, COUNT(*) as n FROM events "
        f"WHERE event='career_impression' AND entity_title IS NOT NULL AND {ts_filter} "
        f"GROUP BY entity_title ORDER BY n DESC LIMIT 10",
        ts_args
    )
    careers_table = "".join(
        f"<tr><td>{r['entity_title']}</td><td class='num'>{r['n']}</td></tr>\n"
        for r in career_rows
    ) or "<tr><td colspan='2' class='empty'>No data</td></tr>"

    # Top courses
    course_rows = q(
        f"SELECT entity_title, COUNT(*) as n FROM events "
        f"WHERE event='course_impression' AND entity_title IS NOT NULL AND {ts_filter} "
        f"GROUP BY entity_title ORDER BY n DESC LIMIT 10",
        ts_args
    )
    courses_table = "".join(
        f"<tr><td>{r['entity_title']}</td><td class='num'>{r['n']}</td></tr>\n"
        for r in course_rows
    ) or "<tr><td colspan='2' class='empty'>No data</td></tr>"

    # API cost
    usage_rows = conn.execute(
        f"SELECT model, SUM(input_tokens) as inp, SUM(output_tokens) as out, SUM(cost_usd) as cost "
        f"FROM api_usage WHERE {ts_filter.replace('ts', 'ts')} GROUP BY model ORDER BY cost DESC",
        ts_args
    ).fetchall()
    total_cost = sum(r["cost"] for r in usage_rows)
    total_calls = conn.execute(
        f"SELECT COUNT(*) as n FROM api_usage WHERE {ts_filter}", ts_args
    ).fetchone()["n"]
    sessions_n = summary["sessions"] or 1
    cost_per_session = total_cost / sessions_n if sessions_n else 0

    if usage_rows:
        cost_rows_html = ""
        for r in usage_rows:
            rates = _MODEL_PRICING.get(r["model"], {"input": 0, "output": 0})
            cost_rows_html += (
                f"<tr><td>{r['model']}</td>"
                f"<td class='num'>{r['inp']:,}</td>"
                f"<td class='num'>{r['out']:,}</td>"
                f"<td class='num'>${r['cost']:.4f}</td></tr>\n"
            )
        cost_section = (
            f'<h2>API cost estimate ({period_label})</h2>'
            f'<p><strong>Total: ${total_cost:.4f}</strong> &nbsp;·&nbsp; '
            f'{total_calls:,} calls &nbsp;·&nbsp; '
            f'${cost_per_session:.4f} per session</p>'
            f'<table><thead><tr><th>Model</th><th class="num">Input tokens</th>'
            f'<th class="num">Output tokens</th><th class="num">Est. cost (USD)</th></tr></thead>'
            f'<tbody>{cost_rows_html}</tbody></table>'
            f'<p class="note">Voyage AI embedding costs not included. Prices based on published Anthropic rates.</p>'
        )
    else:
        cost_section = '<h2>API cost estimate</h2><p class="empty">No API calls recorded yet for this period.</p>'

    # Rate limit events
    rl_rows = q(
        "SELECT ts, meta FROM events WHERE event='rate_limit' ORDER BY ts DESC LIMIT 20"
    )
    if rl_rows:
        rl_table_rows = ""
        for r in rl_rows:
            ts_fmt = r["ts"][:19].replace("T", " ")
            try:
                meta = json.loads(r["meta"] or "{}")
                call_site    = meta.get("call_site", "?")
                retry_after  = meta.get("retry_after", "?")
            except Exception:
                call_site, retry_after = "?", "?"
            rl_table_rows += f"<tr><td>{ts_fmt}</td><td>{call_site}</td><td>{retry_after}s</td></tr>\n"
        rl_section = (
            f'<h2 class="warn">Rate limit events (last 20)</h2>'
            f'<table><thead><tr><th>Time (UTC)</th><th>Call site</th><th>Retry after</th></tr></thead>'
            f'<tbody>{rl_table_rows}</tbody></table>'
        )
    else:
        rl_section = '<h2>Rate limit events</h2><p class="empty">None recorded.</p>'

    period_nav = ""
    for label, val in [("7 days", "7"), ("30 days", "30"), ("All time", "all")]:
        active = ' class="active"' if val == days_param else ""
        period_nav += f'<a href="/admin/analytics?days={val}"{active}>{label}</a> '

    conn.close()

    html = (
        _ANALYTICS_PAGE_HTML
        .replace("{{PERIOD_LABEL}}", period_label)
        .replace("{{PERIOD_NAV}}", period_nav)
        .replace("{{SESSIONS}}", str(summary["sessions"]))
        .replace("{{TOTAL_EVENTS}}", str(summary["total_events"]))
        .replace("{{CHAT_SUBMITS}}", str(chat_submits))
        .replace("{{RATE_LIMIT_COUNT}}", str(rate_limit_count))
        .replace("{{DAILY_CHART}}", daily_chart)
        .replace("{{FUNNEL_ROWS}}", funnel_rows)
        .replace("{{CAREERS_TABLE}}", careers_table)
        .replace("{{COURSES_TABLE}}", courses_table)
        .replace("{{COST_SECTION}}", cost_section)
        .replace("{{RL_SECTION}}", rl_section)
    )
    return make_response(html)


# ---------------------------------------------------------------------------
# Access page HTML
# ---------------------------------------------------------------------------

_ACCESS_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathwayIQ — Enter access code</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f172a;
    color: #f1f5f9;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }
  .card {
    background: #1e293b;
    border-radius: 12px;
    padding: 40px 32px;
    width: 100%;
    max-width: 400px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .brand {
    font-size: 22px;
    font-weight: 600;
    color: #0d9488;
    margin-bottom: 6px;
  }
  .tagline {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 32px;
  }
  label {
    display: block;
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 6px;
  }
  input[type=text] {
    width: 100%;
    padding: 10px 14px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    color: #f1f5f9;
    font-size: 16px;
    outline: none;
    margin-bottom: 16px;
  }
  input[type=text]:focus { border-color: #0d9488; }
  button {
    width: 100%;
    padding: 11px;
    background: #0d9488;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    cursor: pointer;
  }
  button:hover { background: #0f766e; }
  .error {
    margin-bottom: 16px;
    padding: 10px 14px;
    background: #450a0a;
    border: 1px solid #b91c1c;
    border-radius: 8px;
    color: #fca5a5;
    font-size: 13px;
  }
</style>
</head>
<body>
<div class="card">
  <div class="brand">PathwayIQ</div>
  <div class="tagline">Course &amp; Career Explorer</div>
  {{ERROR_BLOCK}}
  <form method="POST" action="/access">
    <label for="code">Enter the access code you were given</label>
    <input type="text" id="code" name="code" placeholder="e.g. maple-7734" autocomplete="one-time-code" autofocus>
    <button type="submit">Continue</button>
  </form>
</div>
</body>
</html>"""

_ACCESS_PAGE_HTML = _ACCESS_PAGE_HTML.replace(
    "{{ERROR_BLOCK}}",
    '<div class="error">{{ERROR}}</div>' if False else "{{ERROR_BLOCK}}"
)

# Rebuild with proper conditional error block handling
def _render_access_page(error=None):
    error_block = f'<div class="error">{error}</div>' if error else ""
    return _ACCESS_PAGE_HTML.replace("{{ERROR_BLOCK}}", error_block).replace("{{ERROR}}", "")


# ---------------------------------------------------------------------------
# Admin page HTML
# ---------------------------------------------------------------------------

_ADMIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathwayIQ Admin — Access Codes</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #1e293b; padding: 32px 24px; }
  h1 { font-size: 20px; color: #0d9488; margin-bottom: 4px; }
  .sub { font-size: 13px; color: #64748b; margin-bottom: 28px; }
  h2 { font-size: 15px; font-weight: 600; margin-bottom: 12px; color: #334155; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 32px; font-size: 14px; }
  th { text-align: left; padding: 8px 12px; background: #e2e8f0; color: #475569; font-weight: 600; }
  td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }
  .active { color: #059669; font-weight: 600; }
  .expired { color: #dc2626; }
  .form-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; margin-bottom: 8px; }
  .field { display: flex; flex-direction: column; gap: 4px; }
  label { font-size: 13px; color: #475569; }
  input[type=text], input[type=number] {
    padding: 8px 12px; border: 1px solid #cbd5e1; border-radius: 6px;
    font-size: 14px; width: 220px; outline: none;
  }
  input[type=number] { width: 100px; }
  input:focus { border-color: #0d9488; }
  button {
    padding: 9px 20px; background: #0d9488; color: #fff; border: none;
    border-radius: 6px; font-size: 14px; cursor: pointer;
  }
  button:hover { background: #0f766e; }
  .hint { font-size: 12px; color: #94a3b8; margin-top: 4px; }
  .btn-revoke {
    padding: 4px 10px; background: #fff; color: #dc2626; border: 1px solid #dc2626;
    border-radius: 4px; font-size: 12px; cursor: pointer;
  }
  .btn-revoke:hover { background: #dc2626; color: #fff; }
</style>
</head>
<body>
<h1>PathwayIQ Admin</h1>
<p class="sub">Access code management</p>

<h2>Current codes</h2>
<table>
  <thead><tr><th>Code</th><th>Label</th><th>Expires</th><th>Used</th><th>Status</th><th></th></tr></thead>
  <tbody>{{TABLE_ROWS}}</tbody>
</table>

{{NEW_CODE}}

<h2>Generate new code</h2>
<form method="POST" action="/admin/codes">
  <div class="form-row">
    <div class="field">
      <label for="label">Label</label>
      <input type="text" id="label" name="label" placeholder="e.g. Claire GMIoT" required>
    </div>
    <div class="field">
      <label for="days">Duration (days)</label>
      <input type="number" id="days" name="days" placeholder="Leave blank = permanent" min="1">
    </div>
    <button type="submit">Generate</button>
  </div>
  <p class="hint">Leave duration blank for a permanent code.</p>
</form>
<p style="margin-top:24px;font-size:13px;color:#64748b;">
  <a href="/admin/analytics" style="color:#0d9488;">View analytics &rarr;</a>
</p>
</body>
</html>"""


_ANALYTICS_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FutureFinder Admin — Analytics</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #1e293b; padding: 32px 24px; max-width: 900px; }
  h1 { font-size: 20px; color: #0d9488; margin-bottom: 4px; }
  h2 { font-size: 15px; font-weight: 600; margin: 28px 0 12px; color: #334155; }
  h2.warn { color: #dc2626; }
  .nav { font-size: 13px; color: #64748b; margin-bottom: 6px; }
  .nav a { color: #0d9488; text-decoration: none; margin-right: 12px; }
  .period { display: flex; gap: 8px; margin-bottom: 28px; }
  .period a {
    padding: 5px 14px; border-radius: 20px; font-size: 13px; text-decoration: none;
    background: #e2e8f0; color: #475569;
  }
  .period a.active { background: #0d9488; color: #fff; }
  .summary { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 8px; }
  .stat { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 24px; min-width: 130px; }
  .stat-val { font-size: 28px; font-weight: 700; color: #0d9488; }
  .stat-label { font-size: 12px; color: #64748b; margin-top: 2px; }
  .stat.warn .stat-val { color: #dc2626; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 8px; font-size: 14px; }
  th { text-align: left; padding: 8px 12px; background: #e2e8f0; color: #475569; font-weight: 600; }
  td { padding: 7px 12px; border-bottom: 1px solid #e2e8f0; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; color: #334155; font-weight: 600; }
  .empty { color: #94a3b8; font-size: 13px; font-style: italic; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
  .bar-label { font-size: 12px; color: #64748b; width: 42px; flex-shrink: 0; }
  .bar-wrap { flex: 1; display: flex; align-items: center; gap: 6px; }
  .bar { height: 16px; background: #0d9488; border-radius: 3px; min-width: 2px; }
  .bar-val { font-size: 12px; color: #475569; }
  p.note { font-size: 12px; color: #94a3b8; margin-top: 6px; }
</style>
</head>
<body>
<div class="nav">
  <a href="/admin/analytics">Analytics</a>
  <a href="/admin/codes">Access codes</a>
</div>
<h1>FutureFinder Analytics</h1>

<div class="period">{{PERIOD_NAV}}</div>

<div class="summary">
  <div class="stat"><div class="stat-val">{{SESSIONS}}</div><div class="stat-label">Sessions</div></div>
  <div class="stat"><div class="stat-val">{{CHAT_SUBMITS}}</div><div class="stat-label">Chat messages</div></div>
  <div class="stat"><div class="stat-val">{{TOTAL_EVENTS}}</div><div class="stat-label">Total events</div></div>
  <div class="stat warn"><div class="stat-val">{{RATE_LIMIT_COUNT}}</div><div class="stat-label">Rate limit hits</div></div>
</div>

<h2>Daily sessions (last 14 days)</h2>
{{DAILY_CHART}}

<h2>Engagement</h2>
<table>
  <thead><tr><th>Event</th><th style="text-align:right">Count</th></tr></thead>
  <tbody>{{FUNNEL_ROWS}}</tbody>
</table>

<div class="two-col">
  <div>
    <h2>Top careers seen</h2>
    <table>
      <thead><tr><th>Career</th><th style="text-align:right">Shown</th></tr></thead>
      <tbody>{{CAREERS_TABLE}}</tbody>
    </table>
  </div>
  <div>
    <h2>Top courses seen</h2>
    <table>
      <thead><tr><th>Course</th><th style="text-align:right">Shown</th></tr></thead>
      <tbody>{{COURSES_TABLE}}</tbody>
    </table>
  </div>
</div>

{{COST_SECTION}}

{{RL_SECTION}}

</body>
</html>"""


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")

@app.get("/")
def serve_index():
    return send_from_directory(APP_DIR, "index.html")

@app.get("/<path:path>")
def serve_static(path):
    return send_from_directory(APP_DIR, path)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/welcome-data")
def api_welcome_data():
    """Return quals, SSA codes, and course-count matrix for the welcome flow.

    quals  — canonical qual_type values that have at least one active course.
    counts — { qual_type: { ssa_code: n } } — qual_type is now the canonical
             vocabulary so no mapping layer is needed.
    """
    conn = sqlite3.connect(FUTUREFINDER_DB)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT DISTINCT CAST(ssa_code AS TEXT) AS ssa
            FROM courses
            WHERE ssa_code IS NOT NULL AND is_active = 1
        """)
        ssa_codes = [r["ssa"] for r in cur.fetchall()]

        cur.execute("""
            SELECT qual_type, CAST(ssa_code AS TEXT) AS ssa, COUNT(*) AS n
            FROM courses
            WHERE qual_type IS NOT NULL AND ssa_code IS NOT NULL AND is_active = 1
            GROUP BY qual_type, ssa_code
        """)
        counts = {}
        for row in cur.fetchall():
            counts.setdefault(row["qual_type"], {})[row["ssa"]] = row["n"]

        quals = list(counts.keys())

        return jsonify({"quals": quals, "ssa_codes": ssa_codes, "counts": counts})
    finally:
        conn.close()


# LEGACY — called by app_v2.js (tile-based UI). Not used by the chat-first frontend.
@app.get("/search/courses")
def search_courses():
    subject       = request.args.get("subject", "").strip()
    q             = request.args.get("q", "").strip()
    qualification = request.args.get("qualification", "").strip()

    # Subject-tile path — direct SQLite lookup by SSA code, no embedding, no limit
    if subject:
        ssa_code = SSA_MAP.get(subject)
        if not ssa_code:
            return jsonify({"subject": subject, "results": [], "message": "Unknown subject area."})
        conn = sqlite3.connect(FUTUREFINDER_DB)
        conn.row_factory = sqlite3.Row
        sql = ("SELECT c.*, p.provider_name FROM courses c "
               "LEFT JOIN providers p ON c.provider_id = p.provider_id "
               "WHERE c.ssa_code = ? AND c.is_active = 1")
        params: list = [ssa_code]
        if qualification:
            sql += " AND c.qual_type = ?"
            params.append(qualification)
        sql += " ORDER BY c.course_title"
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        results = [format_course_from_db(dict(row), 100) for row in rows]
        pass  # qualifying counted in chat() only
        if not results:
            return jsonify({"subject": subject, "results": [],
                            "message": "No courses found for that subject and qualification."})
        built_from = subject + (f" · {qualification}" if qualification else "")
        tile_candidate_set = {
            "course_ids": [r["id"] for r in results],
            "job_ids":    [],
            "built_from": built_from,
        }
        return jsonify({"subject": subject, "results": results,
                        "candidate_set": tile_candidate_set})

    # Semantic search path — used by chat
    if not q:
        return jsonify({"error": "subject or q is required"}), 400

    vector = embed(q)

    hits = match_courses_col.query(
        query_embeddings=[vector],
        n_results=100,
        include=["metadatas", "distances"],
    )

    vector_results = []
    for cid, meta, dist in zip(hits["ids"][0], hits["metadatas"][0], hits["distances"][0]):
        s = score(dist)
        if s >= MIN_SCORE:
            db = ff_course_row(cid)
            if db:
                vector_results.append(format_course_from_db(db, s))

    # Keyword search on course titles — ensures title matches are never buried
    keyword_results = keyword_course_search(q, qualification or None)

    # Merge: keyword matches first, then vector results not already present
    seen_ids = {r["id"] for r in keyword_results}
    merged   = keyword_results + [r for r in vector_results if r["id"] not in seen_ids]

    if not merged:
        return jsonify({"query": q, "results": [], "message": "No courses found matching your query."})
    return jsonify({"query": q, "results": merged})


@app.get("/search/jobs")
def search_jobs():
    q = request.args.get("q", "").strip()

    if not q:
        return jsonify({"error": "q is required"}), 400

    vector = embed(q)

    hits = jobs_col.query(
        query_embeddings=[vector],
        n_results=200,
        where={"chunk": {"$eq": "overview"}},
        include=["metadatas", "distances", "documents"],
    )

    # Fetch paired skills chunks
    matched_ids   = hits["ids"][0]
    skills_ids    = [id_.replace("_overview", "_skills") for id_ in matched_ids]
    skills_lookup = {}
    if skills_ids:
        sk = jobs_col.get(ids=skills_ids, include=["documents"])
        for sk_id, sk_doc in zip(sk["ids"], sk["documents"]):
            skills_lookup[sk_id.replace("_skills", "")] = sk_doc

    results = []
    for id_, meta, dist, ov_doc in zip(
        matched_ids, hits["metadatas"][0], hits["distances"][0], hits["documents"][0]
    ):
        s = score(dist)
        if s >= MIN_SCORE:
            db      = job_row(meta["job_id"])
            job     = format_job(meta, db, s)
            sk_text = skills_lookup.get(str(meta["job_id"]), "")
            job["full_text"] = ov_doc + "\n\n" + sk_text if sk_text else ov_doc
            results.append(job)

    if not results:
        return jsonify({"query": q, "results": [], "message": "No jobs found matching your query."})
    return jsonify({"query": q, "results": results})


# LEGACY — not called by the chat-first frontend. Course detail uses /courses/<id>/detail
# which reads course_career_pathways. This endpoint and connections.db can be removed
# once confirmed no other clients depend on it.
@app.get("/courses/<int:course_id>/careers")
def course_careers(course_id):
    limit = min(int(request.args.get("limit", 3)), 20)

    # --- Connections table fast path ---
    if os.path.exists(CONNECTIONS_DB):
        try:
            cconn = sqlite3.connect(CONNECTIONS_DB)
            rows = cconn.execute(
                """SELECT job_id, semantic_score, skills_score
                   FROM course_job_connections
                   WHERE course_id = ?
                   ORDER BY semantic_score DESC
                   LIMIT ?""",
                (course_id, limit),
            ).fetchall()
            cconn.close()

            if rows:
                # Fetch course name for response
                course_row_db = ff_course_row(str(course_id))
                course_name   = course_row_db["course_title"] if course_row_db else None

                results = []
                for job_id, semantic_score, skills_score in rows:
                    jid  = str(job_id)
                    db   = job_row(jid)
                    meta_hit = jobs_col.get(
                        ids=[f"{jid}_overview"], include=["metadatas"]
                    )
                    if not meta_hit["ids"]:
                        continue
                    meta = meta_hit["metadatas"][0]
                    job  = format_job(meta, db, semantic_score)
                    job["skills_score"] = skills_score
                    caution = (
                        (semantic_score - skills_score) > CAUTION_DIVERGENCE_THRESHOLD
                        if skills_score is not None else False
                    )
                    job["caution"] = caution
                    results.append(job)

                return jsonify({
                    "course_id":   course_id,
                    "course_name": course_name,
                    "source":      "connections_table",
                    "results":     results,
                })
        except Exception as e:
            print(f"[connections] table lookup failed ({e}) — falling back to live search", flush=True)

    # --- Live search fallback ---
    # Lift the stored vector from match_courses and query against jobs
    stored = match_courses_col.get(
        ids=[str(course_id)],
        include=["embeddings", "metadatas"],
    )
    if not stored["ids"]:
        return jsonify({"error": f"Course {course_id} not found in index"}), 404

    vector = stored["embeddings"][0]
    course_meta = stored["metadatas"][0]

    # Query against overview chunks — scores reflect duties similarity
    hits = jobs_col.query(
        query_embeddings=[vector],
        n_results=limit,
        where={"chunk": {"$eq": "overview"}},
        include=["metadatas", "distances", "documents"],
    )

    # Fetch paired skills chunks for all matched jobs
    matched_ids = hits["ids"][0]
    skills_ids  = [id_.replace("_overview", "_skills") for id_ in matched_ids]
    skills_lookup = {}
    if skills_ids:
        sk = jobs_col.get(ids=skills_ids, include=["documents"])
        for sk_id, sk_doc in zip(sk["ids"], sk["documents"]):
            skills_lookup[sk_id.replace("_skills", "")] = sk_doc

    results = []
    for id_, meta, dist, ov_doc in zip(
        matched_ids, hits["metadatas"][0], hits["distances"][0], hits["documents"][0]
    ):
        s = score(dist)
        if s >= MIN_SCORE:
            jid          = str(meta["job_id"])
            sk_text      = skills_lookup.get(jid, "")
            db           = job_row(jid)
            job          = format_job(meta, db, s)
            job["full_text"] = ov_doc + "\n\n" + sk_text if sk_text else ov_doc

            if s < CROSS_COLLECTION_MIN_DOMAIN:
                print(f"[caution] course {course_id} -> job {jid}: domain={s}% EXCLUDED (domain below floor)", flush=True)
                continue

            skills_score = compute_skills_score(course_id, jid)
            sk_pct       = f"{skills_score}%" if skills_score is not None else "N/A"

            if skills_score is None or skills_score < CROSS_COLLECTION_MIN_SKILLS:
                print(f"[caution] course {course_id} -> job {jid}: domain={s}% skills={sk_pct} EXCLUDED (skills below floor)", flush=True)
                continue

            caution = (s - skills_score) > CAUTION_DIVERGENCE_THRESHOLD if skills_score is not None else False
            job["skills_score"] = skills_score
            job["caution"]      = caution

            flag = "FLAGGED" if caution else "ok"
            print(f"[caution] course {course_id} -> job {jid}: domain={s}% skills={sk_pct} d={s - (skills_score or s):+d}% {flag}", flush=True)

            results.append(job)

    return jsonify({
        "course_id":   course_id,
        "course_name": course_meta.get("course_name"),
        "results":     results,
    })


@app.get("/jobs/<int:job_id>/courses")
def job_courses(job_id):
    limit = min(int(request.args.get("limit", 5)), 20)

    # --- Cache check ---
    _jpc = sqlite3.connect(JOBS_DB)
    _jpc.row_factory = sqlite3.Row
    cached_row = _jpc.execute(
        "SELECT courses_json, courses_cache_version FROM job_progression_cache "
        "WHERE job_id = ? AND courses_json IS NOT NULL",
        (job_id,)
    ).fetchone()
    if cached_row and cached_row["courses_cache_version"] == COURSES_CACHE_VERSION:
        _jpc.close()
        print(f"[job_courses] job_id={job_id} cache hit (v{COURSES_CACHE_VERSION})", flush=True)
        selected_ids = json.loads(cached_row["courses_json"])
        results = []
        for cid in selected_ids[:limit]:
            db = ff_course_row(str(cid))
            if db:
                course = format_course_from_db(db, None)
                if course:
                    results.append(course)
        j = job_row(str(job_id))
        return jsonify({"job_id": job_id, "job_title": j["title"] if j else "", "results": results})
    _jpc.close()

    # Lift the skills chunk vector from the jobs collection
    stored = jobs_col.get(
        ids=[f"{job_id}_skills"],
        include=["embeddings", "metadatas"],
    )
    if not stored["ids"]:
        stored = jobs_col.get(
            ids=[f"{job_id}_overview"],
            include=["embeddings", "metadatas"],
        )
    if not stored["ids"]:
        return jsonify({"error": f"Job {job_id} not found in index"}), 404

    vector   = stored["embeddings"][0]
    job_meta = stored["metadatas"][0]

    # Fetch a larger pool so Haiku has enough to choose from
    hits = match_courses_col.query(
        query_embeddings=[vector],
        n_results=15,
        include=["metadatas", "distances"],
    )

    candidates = []
    for cid, meta, dist in zip(hits["ids"][0], hits["metadatas"][0], hits["distances"][0]):
        s = score(dist)
        db = ff_course_row(cid)
        if db:
            candidates.append({"db": db, "score": s, "cid": cid})

    if not candidates:
        return jsonify({"job_id": job_id, "job_title": job_meta.get("title"), "results": []})

    # Haiku ratification — select the genuinely relevant courses for this job
    job = job_row(str(job_id))
    job_context = ""
    if job:
        job_context = (
            f"Job title: {job['title']}\n"
            f"Skills required: {(job.get('skills_required') or '')[:400]}\n"
            f"Entry routes: {(job.get('entry_routes') or '')[:300]}"
        )
    else:
        job_context = f"Job title: {job_meta.get('title', '')}"

    candidate_lines = [
        f"{c['cid']} | {c['db']['course_title']} | {c['db'].get('qual_type','')} Level {c['db'].get('level','')} | {(c['db'].get('overview') or '')[:200]}"
        for c in candidates
    ]
    haiku_msg = (
        f"{job_context}\n\n"
        f"Candidate courses (ID | Title | Qual Level | Overview):\n"
        + "\n".join(candidate_lines)
        + f"\n\nSelect the {limit} courses that most genuinely prepare someone for this role. "
        f"Read the job's skills and entry routes carefully — only include courses whose content "
        f"clearly aligns with what this job actually requires."
    )

    selected_ids = None
    try:
        resp = _anthropic_post({
            "model":       HAIKU_MODEL,
            "max_tokens":  300,
            "temperature": 0.2,
            "system": (
                "You are matching courses to a job role. "
                "Read the job description carefully and select only courses that "
                "genuinely prepare someone for that specific role. "
                "Reject courses that are only superficially related."
            ),
            "tools":       [_SELECT_COURSES_TOOL],
            "tool_choice": {"type": "tool", "name": "select_courses"},
            "messages":    [{"role": "user", "content": haiku_msg}],
        }, call_site="job_courses_ratify", timeout=20.0)
        tool_use = next(
            (b for b in resp.json()["content"] if b["type"] == "tool_use"), None
        )
        if tool_use:
            selected_ids = [str(i) for i in (tool_use["input"].get("selected_course_ids") or [])]  # strings to match Chroma cid keys
            print(f"[job_courses] Haiku selected {len(selected_ids)} courses for job {job_id}", flush=True)
    except RateLimitError:
        print(f"[job_courses] rate limited — falling back to Chroma top {limit}", flush=True)
    except Exception as e:
        print(f"[job_courses] Haiku error: {e} — falling back to Chroma top {limit}", flush=True)

    # Build results — Haiku order if available, else top Chroma hits
    cand_by_id = {c["cid"]: c for c in candidates}
    if selected_ids:
        ordered = [cand_by_id[sid] for sid in selected_ids if sid in cand_by_id]
    else:
        ordered = candidates[:limit]

    results = []
    for c in ordered[:limit]:
        course = format_course_from_db(c["db"], c["score"])
        if course:
            results.append(course)

    # Cache the ordered course IDs for future requests
    if selected_ids or ordered:
        ids_to_cache = selected_ids if selected_ids else [c["cid"] for c in ordered]
        try:
            _jpc = sqlite3.connect(JOBS_DB)
            existing = _jpc.execute(
                "SELECT job_id FROM job_progression_cache WHERE job_id = ?", (job_id,)
            ).fetchone()
            if existing:
                _jpc.execute(
                    "UPDATE job_progression_cache SET courses_json = ?, courses_cache_version = ? WHERE job_id = ?",
                    (json.dumps(ids_to_cache), COURSES_CACHE_VERSION, job_id)
                )
            else:
                _jpc.execute(
                    "INSERT INTO job_progression_cache (job_id, courses_json, courses_cache_version) VALUES (?, ?, ?)",
                    (job_id, json.dumps(ids_to_cache), COURSES_CACHE_VERSION)
                )
            _jpc.commit()
            _jpc.close()
            print(f"[job_courses] job_id={job_id} cached {len(ids_to_cache)} course IDs", flush=True)
        except Exception as e:
            print(f"[job_courses] cache write failed ({e})", flush=True)

    return jsonify({
        "job_id":    job_id,
        "job_title": job_meta.get("title"),
        "results":   results,
    })


@app.get("/courses/<int:course_id>/detail")
def course_detail_ff(course_id):
    """Full course detail from futurefinder.sqlite + career pathways + card job titles."""
    conn = sqlite3.connect(FUTUREFINDER_DB)
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT c.*, p.provider_name "
        "FROM courses c "
        "LEFT JOIN providers p ON c.provider_id = p.provider_id "
        "WHERE c.course_id = ? AND c.is_active = 1",
        (course_id,),
    ).fetchone()

    if not row:
        conn.close()
        return jsonify({"error": f"Course {course_id} not found"}), 404

    pathways_row = conn.execute(
        "SELECT narrative_short, narrative, career_narrative, "
        "card_jobs, curated_jobs, node_groups "
        "FROM course_career_pathways WHERE course_id = ?",
        (course_id,),
    ).fetchone()
    conn.close()

    # Resolve card_job IDs → titles from jobs DB
    card_jobs_out = []
    if pathways_row and pathways_row["card_jobs"]:
        try:
            job_ids = json.loads(pathways_row["card_jobs"])
            if job_ids:
                jconn = sqlite3.connect(JOBS_DB)
                jconn.row_factory = sqlite3.Row
                placeholders = ",".join("?" * len(job_ids))
                job_rows = jconn.execute(
                    f"SELECT id, title FROM jobs WHERE id IN ({placeholders})",
                    job_ids,
                ).fetchall()
                jconn.close()
                id_to_title = {r["id"]: r["title"] for r in job_rows}
                card_jobs_out = [
                    {"job_id": jid, "title": id_to_title.get(jid, "")}
                    for jid in job_ids
                ]
        except Exception as e:
            print(f"[course_detail_ff] card_jobs parse error: {e}", flush=True)

    curated_jobs_out = []
    if pathways_row and pathways_row["curated_jobs"]:
        try:
            curated_jobs_out = json.loads(pathways_row["curated_jobs"])
        except Exception as e:
            print(f"[course_detail_ff] curated_jobs parse error: {e}", flush=True)

    return jsonify({
        "course_id":          row["course_id"],
        "course_title":       row["course_title"],
        "provider":           row["provider_name"] or "",
        "campus":             row["campus"] or "",
        "qual_type":          row["qual_type"] or "",
        "level":              row["level"],
        "mode":               row["mode"] or "",
        "duration":           row["duration"] or "",
        "course_url":         row["course_url"] or "",
        "preview":            row["preview"] or "",
        "overview":           row["overview"] or "",
        "content":            row["content"] or "",
        "entry_requirements": row["entry_requirements"] or "",
        "progression":        row["progression"] or "",
        "pathways": {
            "narrative_short":   pathways_row["narrative_short"]   if pathways_row else "",
            "career_narrative":   pathways_row["career_narrative"]  if pathways_row else None,
            "card_jobs":         card_jobs_out,
            "curated_jobs":      curated_jobs_out,
            "node_groups":       json.loads(pathways_row["node_groups"])
                                 if pathways_row and pathways_row["node_groups"] else None,
        } if pathways_row else None,
    })


@app.get("/courses/<int:course_id>")
def course_detail(course_id):
    db = ff_course_row(str(course_id))
    if not db:
        return jsonify({"error": f"Course {course_id} not found"}), 404
    return jsonify({
        "id":                  db["course_id"],
        "title":               db["course_title"],
        "provider":            db.get("provider_name") or "",
        "level":               db.get("level"),
        "qual_type":           db.get("qual_type"),
        "mode":                db.get("mode"),
        "campus":              db.get("campus") or "",
        "course_url":          db.get("course_url"),
        "ssa_code":            db.get("ssa_code"),
        "overview":            db.get("overview") or "",
        "content":             db.get("content") or "",
        "entry_requirements":  db.get("entry_requirements") or "",
        "progression":         db.get("progression") or "",
    })


@app.get("/jobs/<int:job_id>")
def job_detail(job_id):
    db = job_row(str(job_id))
    if not db:
        return jsonify({"error": f"Job {job_id} not found"}), 404
    sal = salary_string(db.get("salary_min"), db.get("salary_max"),
                        db.get("salary_currency", "GBP"))
    return jsonify({
        "id":                  db["id"],
        "title":               db["title"],
        "source":              (db.get("source") or "").upper(),
        "source_url":          db.get("url"),
        "salary_min":          db.get("salary_min"),
        "salary_max":          db.get("salary_max"),
        "salary_display":      sal,
        "overview":            db.get("overview") or "",
        "typical_duties":      db.get("typical_duties") or "",
        "skills_required":     db.get("skills_required") or "",
        "entry_routes":        db.get("entry_routes") or "",
        "salary":              db.get("salary") or "",
        "career_progression":  db.get("progression") or "",
        "has_progression":     bool(db.get("overview")),
        "employer_text":       lmi_employer_text(job_id),
    })


@app.get("/jobs/<int:job_id>/progression")
def job_progression(job_id):
    jobs_conn = sqlite3.connect(JOBS_DB)
    jobs_conn.row_factory = sqlite3.Row

    # Step 1 — Check cache
    cached = jobs_conn.execute(
        "SELECT narrative, inbound_json, outbound_json FROM job_progression_cache "
        "WHERE job_id = ? AND prompt_version = 5", (job_id,)
    ).fetchone()
    if cached:
        jobs_conn.close()
        print(f"[progression] job_id={job_id} cache hit", flush=True)
        return jsonify({
            "has_progression": True,
            "cached":          True,
            "narrative":       cached["narrative"],
            "inbound":         json.loads(cached["inbound_json"]),
            "outbound":        json.loads(cached["outbound_json"]),
        })

    # Step 2 — Get current job profile
    job = jobs_conn.execute(
        "SELECT id, title, overview, typical_duties, skills_required, entry_routes, progression, career_prospects "
        "FROM jobs WHERE id = ?", (str(job_id),)
    ).fetchone()
    if not job or not job["overview"]:
        jobs_conn.close()
        return jsonify({"has_progression": False})
    job = dict(job)

    # Step 3 — Candidate jobs via Chroma cross-collection search
    stored_vec = get_stored_vector(jobs_col, f"{job_id}_overview")
    if stored_vec is None:
        jobs_conn.close()
        return jsonify({"has_progression": False})

    hits = jobs_col.query(
        query_embeddings=[stored_vec],
        n_results=35,
        where={"chunk": {"$eq": "overview"}},
        include=["metadatas"],
    )

    candidate_ids = []
    for meta in hits["metadatas"][0]:
        jid = int(meta["job_id"])
        if jid != job_id and jid not in candidate_ids:
            candidate_ids.append(jid)
        if len(candidate_ids) >= 30:
            break

    candidates = []
    for cid in candidate_ids:
        row = jobs_conn.execute(
            "SELECT id, title, overview, typical_duties FROM jobs WHERE id = ?", (str(cid),)
        ).fetchone()
        if row:
            candidates.append({
                "id":             row["id"],
                "title":          row["title"],
                "overview":       (row["overview"] or "")[:150],
                "typical_duties": (row["typical_duties"] or "")[:150],
            })

    # Step 4 — Build Sonnet prompt
    candidate_block = "\n\n---\n\n".join(
        f"ID: {c['id']}\nTitle: {c['title']}\n"
        f"Overview: {c['overview']}\nTypical duties: {c['typical_duties']}"
        for c in candidates
    )
    user_prompt = (
        f"Here is a job profile:\n\n"
        f"Title: {job['title']}\n"
        f"Overview: {job['overview']}\n"
        f"Typical duties: {job['typical_duties']}\n"
        f"Skills required: {job['skills_required']}\n\n"
        f"ENTRY ROUTES (from NCS/Prospects career experts):\n"
        f"{job['entry_routes']}\n\n"
        f"CAREER PROGRESSION (from NCS/Prospects career experts):\n"
        f"{job['career_prospects'] or job['progression']}\n\n"
        f"Here are {len(candidates)} candidate job profiles from our database:\n\n"
        f"{candidate_block}\n\n"
        f"Your task:\n"
        f"1. Identify up to 4 candidates that someone might typically come FROM before reaching "
        f"this role — roles that naturally lead here, usually at a lower seniority level. "
        f"Use the entry routes above to guide your selection. "
        f"Only include roles that are a genuinely close fit. Fewer strong connections are better than "
        f"padding the list with weak ones. If this is an entry-level role, there may be no natural "
        f"preceding roles — return an empty inbound array rather than forcing connections.\n"
        f"2. Identify up to 4 candidates this role might naturally progress TO — roles at a "
        f"higher seniority or broader responsibility level. "
        f"Use the career progression above to guide your selection. "
        f"Only include roles that are a genuinely close fit. Fewer strong connections are better than "
        f"padding the list with weak ones. If this is a senior or specialist role near the top of its "
        f"field, there may be no natural outbound roles — return an empty outbound array rather than "
        f"forcing connections.\n"
        f"3. Write 2–3 sentences of warm, plain-English guidance explaining the progression "
        f"landscape for this role, suitable for a college student considering their future career. "
        f"Draw on the specific routes, qualifications, and next steps in the career progression "
        f"above — use their detail to make the narrative specific and grounded. "
        f"Do not name or cite the sources (NCS, Prospects, career experts) — just use the information. "
        f"Keep it practical and directly relevant to this role.\n\n"
        f"Only select candidates from the list provided. If no candidates fit naturally as "
        f"inbound or outbound, return an empty array for that direction — do not force connections.\n\n"
        f'Respond with this JSON structure only:\n'
        f'{{"narrative": "...", "inbound": [{{"id": 42, "title": "..."}}], "outbound": [{{"id": 17, "title": "..."}}]}}'
    )

    print(f"[progression] job_id={job_id} title={job['title']!r} candidates={len(candidates)}", flush=True)

    # Step 5 — Call Sonnet
    try:
        resp = _anthropic_post({
            "model":      SONNET_MODEL,
            "max_tokens": 1500,
            "system":     PROGRESSION_SYSTEM_PROMPT,
            "messages":   [{"role": "user", "content": user_prompt}],
        }, call_site="progression", timeout=60.0)
        result_text = resp.json()["content"][0]["text"].strip()
        # Strip markdown code fences if present
        if result_text.startswith("```"):
            result_text = result_text[result_text.find("\n")+1:]
            if result_text.endswith("```"):
                result_text = result_text[:-3].rstrip()
        result = json.loads(result_text)
    except RateLimitError:
        jobs_conn.close()
        return jsonify({"has_progression": False})
    except Exception as e:
        print(f"[progression] Sonnet call failed ({e})", flush=True)
        jobs_conn.close()
        return jsonify({"has_progression": False})

    # Step 6 — Write to cache
    try:
        jobs_conn.execute(
            "INSERT OR REPLACE INTO job_progression_cache "
            "(job_id, narrative, inbound_json, outbound_json, prompt_version, created_at) "
            "VALUES (?, ?, ?, ?, 5, ?)",
            (job_id,
             result["narrative"],
             json.dumps(result.get("inbound", [])),
             json.dumps(result.get("outbound", [])),
             time.strftime("%Y-%m-%dT%H:%M:%S"))
        )
        jobs_conn.commit()
    except Exception as e:
        print(f"[progression] cache write failed ({e})", flush=True)
    jobs_conn.close()

    print(f"[progression] inbound={len(result.get('inbound',[]))} outbound={len(result.get('outbound',[]))}", flush=True)

    # Step 7 — Return
    return jsonify({
        "has_progression": True,
        "cached":          False,
        "narrative":       result["narrative"],
        "inbound":         result.get("inbound", []),
        "outbound":        result.get("outbound", []),
    })


@app.get("/jobs/<int:job_id>/explain")
def job_explain(job_id):
    jobs_conn = sqlite3.connect(JOBS_DB)
    jobs_conn.row_factory = sqlite3.Row

    # Check cache
    cached = jobs_conn.execute(
        "SELECT explain_text, explain_cache_version FROM job_progression_cache "
        "WHERE job_id = ? AND explain_text IS NOT NULL",
        (job_id,)
    ).fetchone()
    if cached and cached["explain_cache_version"] == EXPLAIN_CACHE_VERSION:
        jobs_conn.close()
        print(f"[explain] job_id={job_id} cache hit (v{EXPLAIN_CACHE_VERSION})", flush=True)
        return jsonify({"text": cached["explain_text"]})

    # Get job title
    job = jobs_conn.execute(
        "SELECT id, title FROM jobs WHERE id = ?", (str(job_id),)
    ).fetchone()
    if not job:
        jobs_conn.close()
        return jsonify({"error": "Job not found"}), 404

    title = job["title"]
    query = (
        f"Tell me about the career pathway for {title} — what the role involves, "
        f"how people typically get into it, what qualifications or experience help, "
        f"and where it can lead."
    )

    print(f"[explain] job_id={job_id} title={title!r} — calling Haiku", flush=True)
    text = chat_explain(query, [], max_tokens=600)

    # Cache against existing progression cache row if present, otherwise insert
    existing = jobs_conn.execute(
        "SELECT job_id FROM job_progression_cache WHERE job_id = ?", (job_id,)
    ).fetchone()
    try:
        if existing:
            jobs_conn.execute(
                "UPDATE job_progression_cache SET explain_text = ?, explain_cache_version = ? WHERE job_id = ?",
                (text, EXPLAIN_CACHE_VERSION, job_id)
            )
        else:
            jobs_conn.execute(
                "INSERT INTO job_progression_cache (job_id, explain_text, explain_cache_version) VALUES (?, ?, ?)",
                (job_id, text, EXPLAIN_CACHE_VERSION)
            )
        jobs_conn.commit()
    except Exception as e:
        print(f"[explain] cache write failed ({e})", flush=True)

    jobs_conn.close()
    return jsonify({"text": text})


@app.post("/saved/campuses")
def saved_campuses():
    """Return deduplicated campus data (with coordinates) for a list of saved course IDs.

    Each entry: {provider, campus_name, postcode, lat, lng, courses: [title, ...]}.
    Matches course.campus to campuses.campus_name by substring; falls back to
    the provider's Main Campus, then the provider's first campus.
    """
    body       = request.get_json(force=True) or {}
    course_ids = body.get("course_ids", [])
    if not course_ids:
        return jsonify([])

    try:
        conn = sqlite3.connect(FUTUREFINDER_DB)
        conn.row_factory = sqlite3.Row

        placeholders = ",".join("?" * len(course_ids))
        courses = conn.execute(
            f"SELECT c.course_id, c.course_title, c.campus, c.provider_id, p.provider_name "
            f"FROM courses c JOIN providers p ON c.provider_id = p.provider_id "
            f"WHERE c.course_id IN ({placeholders})",
            course_ids,
        ).fetchall()

        # Load all campuses for relevant providers
        provider_ids = list({c["provider_id"] for c in courses})
        p_placeholders = ",".join("?" * len(provider_ids))
        all_campuses = conn.execute(
            f"SELECT campus_id, provider_id, campus_name, postcode, lat, lng "
            f"FROM campuses WHERE provider_id IN ({p_placeholders})",
            provider_ids,
        ).fetchall()
        conn.close()
    except Exception as e:
        print(f"[saved/campuses] error: {e}", flush=True)
        return jsonify([])

    # Index campuses by provider_id
    by_provider: dict[int, list] = {}
    for cam in all_campuses:
        by_provider.setdefault(cam["provider_id"], []).append(cam)

    def best_campus(course_campus: str, provider_id: int):
        candidates = by_provider.get(provider_id, [])
        if not candidates:
            return None
        if course_campus:
            cc = course_campus.lower()
            # substring match either way
            for cam in candidates:
                cn = cam["campus_name"].lower()
                if cn in cc or cc in cn:
                    return cam
        # fallback: Main Campus, then first
        for cam in candidates:
            if cam["campus_name"].lower() == "main campus":
                return cam
        return candidates[0]

    # Map campus_id → {meta, courses[]}
    campus_map: dict[int, dict] = {}
    for course in courses:
        cam = best_campus(course["campus"], course["provider_id"])
        if not cam:
            continue
        cid = cam["campus_id"]
        if cid not in campus_map:
            campus_map[cid] = {
                "provider":    course["provider_name"],
                "campus_name": cam["campus_name"],
                "postcode":    cam["postcode"],
                "lat":         cam["lat"],
                "lng":         cam["lng"],
                "courses":     [],
            }
        if course["course_title"] not in campus_map[cid]["courses"]:
            campus_map[cid]["courses"].append(course["course_title"])

    return jsonify(list(campus_map.values()))


def get_sample_courses() -> dict:
    """One random active course per SSA code — used for the 'Show me some ideas' sampler."""
    try:
        conn = sqlite3.connect(FUTUREFINDER_DB)
        conn.row_factory = sqlite3.Row
        codes = [r[0] for r in conn.execute(
            'SELECT DISTINCT ssa_code FROM courses WHERE ssa_code IS NOT NULL AND is_active=1 ORDER BY ssa_code'
        ).fetchall()]
        courses = []
        for code in codes:
            row = conn.execute(
                'SELECT course_id, course_title, preview FROM courses '
                'WHERE ssa_code=? AND is_active=1 ORDER BY RANDOM() LIMIT 1',
                (code,)
            ).fetchone()
            if row:
                courses.append({
                    "course_id":    row["course_id"],
                    "course_title": row["course_title"],
                    "preview_text": (row["preview"] or "")[:200].rstrip(),
                })
        conn.close()
    except Exception as e:
        print(f"[sample_courses] error: {e}", flush=True)
        return {"intro_text": "Here are some courses from across our subject areas.", "courses": []}
    return {
        "intro_text": "Here's a taster — one course from each subject area at GMIoT.",
        "courses": courses,
    }


def get_filtered_courses(ssa_code: int) -> dict:
    """All active courses for a given SSA code, sorted by title."""
    try:
        conn = sqlite3.connect(FUTUREFINDER_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT course_id, course_title, preview FROM courses '
            'WHERE ssa_code=? AND is_active=1 ORDER BY course_title',
            (ssa_code,)
        ).fetchall()
        conn.close()
        courses = [
            {
                "course_id":    r["course_id"],
                "course_title": r["course_title"],
                "preview_text": (r["preview"] or "")[:200].rstrip(),
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[filtered_courses] ssa_code={ssa_code} error: {e}", flush=True)
        return {"intro_text": "Here are the courses in that area.", "courses": []}
    return {"intro_text": "Here are all the courses in that subject area at GMIoT.", "courses": courses}


def retrieve_courses_for_pivot(session_id: str, saved_items: list | None = None) -> dict:
    """
    Build a semantic query from the user's welcome conversation turns, embed it,
    retrieve top Chroma course candidates, then use Haiku to select the best 5-8.
    Returns {"intro_text": str, "courses": [{"course_id", "course_title", "preview_text"}]}.
    """
    saved_items = saved_items or []
    _empty = {
        "intro_text": "I couldn't find good matches yet — try telling me a bit more about what interests you.",
        "courses": [],
    }

    sess = get_welcome_session(session_id)
    with _welcome_sessions_lock:
        messages = list(sess["messages"])

    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    if not user_turns:
        return _empty

    query_text = " ".join(user_turns)
    print(f"[pivot_retrieval] session={session_id[:8]}... query={query_text!r}", flush=True)

    # Embed the user's stated interests
    try:
        embed_result = vo.embed(
            [query_text], model=VOYAGE_MODEL,
            input_type="query", output_dimension=VOYAGE_DIMS,
        )
        vector = embed_result.embeddings[0]
    except Exception as e:
        print(f"[pivot_retrieval] embed error: {e}", flush=True)
        return _empty

    # Chroma — match_courses collection (one chunk per course, futurefinder IDs)
    try:
        hits = match_courses_col.query(
            query_embeddings=[vector],
            n_results=25,
            include=["metadatas", "distances", "documents"],
        )
    except Exception as e:
        print(f"[pivot_retrieval] Chroma error: {e}", flush=True)
        return _empty

    candidates = []
    for cid, meta, dist, doc in zip(
        hits["ids"][0], hits["metadatas"][0], hits["distances"][0], hits["documents"][0]
    ):
        candidates.append({
            "course_id": int(cid),
            "title":     meta.get("title", ""),
            "qual_type": meta.get("qual_type", ""),
            "level":     meta.get("level", ""),
            "preview":   (doc or "")[:300],
        })

    if not candidates:
        return _empty

    # Format conversation context for Haiku
    conv_lines = []
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        conv_lines.append(f"{role}: {m['content']}")
    conversation_text = "\n".join(conv_lines)

    saved_section = ""
    if saved_items:
        saved_titles = ", ".join(i["title"] for i in saved_items if i.get("title"))
        if saved_titles:
            saved_section = f"\n\nSaved items (do not re-recommend these): {saved_titles}"

    candidate_lines = [
        f"{c['course_id']} | {c['title']} | {c['qual_type']} Level {c['level']} | {c['preview']}"
        for c in candidates
    ]
    haiku_msg = (
        f"Conversation:\n{conversation_text}"
        + saved_section
        + f"\n\nCandidate courses (ID | Title | Qual Level | Preview):\n"
        + "\n".join(candidate_lines)
        + "\n\nUsing the conversation above as your guide, select the 5–8 courses that best match what this specific user has said they want. The conversation is your primary input — read it carefully before selecting."
    )

    # Haiku tool-use selection
    try:
        resp = _anthropic_post({
            "model":       HAIKU_MODEL,
            "max_tokens":  300,
            "temperature": 0.3,
            "system": _FF_BASE_SYSTEM + """

## Your task

Select 5–8 courses from the candidate list that best match what this specific
student has said they want. The full conversation is your primary guide —
read it from the beginning, not just the most recent messages.

- Respect the subject the student established early in the conversation even
  if later messages focus on level or qualification type.
- If the student mentions a combination of interests, prioritise courses that
  span both over courses that cover only one.
- Use the qual_type and level shown for each candidate to match the student's
  situation (e.g. someone with only GCSEs cannot yet enter a Level 4 course
  without a Level 3 first; someone who already has a degree should see
  postgraduate or higher-level options).
- If fewer than 5 candidates genuinely match the student's subject, return only
  the ones that do. Do not pad with unrelated courses to hit a minimum count —
  3 good matches is better than 6 with fillers. If there is a level gap, note
  it in intro_text.""",
            "tools":       [_SELECT_COURSES_TOOL],
            "tool_choice": {"type": "tool", "name": "select_courses"},
            "messages":    [{"role": "user", "content": haiku_msg}],
        }, call_site="pivot_retrieval", timeout=20.0)
        tool_use = next(
            (b for b in resp.json()["content"] if b["type"] == "tool_use"), None
        )
        if not tool_use:
            raise ValueError("no tool_use block in response")

        selected_ids = [str(i) for i in (tool_use["input"].get("selected_course_ids") or [])]
        intro_text   = tool_use["input"].get("intro_text") or "Here are some courses that might interest you."
        print(f"[pivot_retrieval] selected={selected_ids} intro={intro_text!r}", flush=True)

    except RateLimitError:
        selected_ids = [str(c["course_id"]) for c in candidates[:5]]
        intro_text   = "Here are some courses that might interest you."
    except Exception as e:
        print(f"[pivot_retrieval] Haiku error: {e} — falling back to top 5 Chroma hits", flush=True)
        selected_ids = [str(c["course_id"]) for c in candidates[:5]]
        intro_text   = "Here are some courses that might interest you."

    # Batch-fetch overview from SQLite for final display text
    id_to_cand = {str(c["course_id"]): c for c in candidates}
    int_ids    = [int(i) for i in selected_ids if i in id_to_cand]

    db_map = {}
    if int_ids:
        placeholders = ",".join("?" * len(int_ids))
        try:
            conn = sqlite3.connect(FUTUREFINDER_DB)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT course_id, course_title, preview FROM courses "
                f"WHERE course_id IN ({placeholders})",
                int_ids,
            ).fetchall()
            conn.close()
            db_map = {str(r["course_id"]): r for r in rows}
        except Exception as e:
            print(f"[pivot_retrieval] SQLite fetch error: {e}", flush=True)

    courses_out = []
    for cid in selected_ids:
        if cid not in id_to_cand:
            continue
        cand    = id_to_cand[cid]
        db_row  = db_map.get(cid)
        title   = db_row["course_title"] if db_row else cand["title"]
        preview = ((db_row["preview"] or "") if db_row else cand["preview"])[:200].rstrip()
        courses_out.append({
            "course_id":    int(cid),
            "course_title": title,
            "preview_text": preview,
        })

    return {"intro_text": intro_text, "courses": courses_out}


@app.post("/chat/welcome")
def chat_welcome():
    cleanup_welcome_sessions()
    body        = request.get_json(force=True) or {}
    message     = (body.get("message") or "").strip()
    session_id  = (body.get("session_id") or "").strip()
    saved_items = body.get("saved_items") or []

    if not message:
        return jsonify({"error": "message is required"}), 400
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    # Keyword shortcut — bypass Sonnet, return one course per SSA area
    if message.lower() == "show me some ideas":
        return jsonify({
            "session_id":       session_id,
            "bot_response":     "Here's a taster of what GMIoT has to offer — one course from each subject area. See anything that appeals?",
            "pivot_to_courses": True,
            "course_list":      get_sample_courses(),
        })

    result = welcome_chat_llm(session_id, message, saved_items)
    if result["bot_response"] is None:
        return jsonify({"error": "llm_error"}), 502

    pivot       = result["pivot_to_courses"]
    course_list = None
    if result.get("filter_code"):
        course_list = get_filtered_courses(result["filter_code"])
        pivot = True
    elif pivot:
        course_list = retrieve_courses_for_pivot(session_id, saved_items)

    if pivot:
        sess = get_welcome_session(session_id)
        with _welcome_sessions_lock:
            sess["pivot_done"] = True

    return jsonify({
        "session_id":       session_id,
        "bot_response":     result["bot_response"],
        "pivot_to_courses": pivot,
        "course_list":      course_list,
        "suggestions":      result.get("suggestions") or [],
        "show_qual_map":    result.get("show_qual_map") or False,
    })




# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def _init_analytics_db():
    """Create analytics.db tables if they don't exist (first-run on fresh deploy)."""
    try:
        conn = sqlite3.connect(ANALYTICS_DB)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, ts TEXT NOT NULL, event TEXT NOT NULL, "
            "entity_type TEXT, entity_id INTEGER, entity_title TEXT, meta TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS access_codes ("
            "code TEXT PRIMARY KEY, "
            "label TEXT NOT NULL, "
            "expires_at TEXT, "
            "created_at TEXT NOT NULL, "
            "used_count INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS api_usage ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL, "
            "session_id TEXT, "
            "call_site TEXT NOT NULL, "
            "model TEXT NOT NULL, "
            "input_tokens INTEGER NOT NULL, "
            "output_tokens INTEGER NOT NULL, "
            "cost_usd REAL NOT NULL)"
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

_init_analytics_db()

# Seed access codes from environment if set.
# SEED_ACCESS_CODES: JSON array of {code, label, expires_at} objects — generated by
# scripts/sync_access_codes.py before each push. INSERT OR IGNORE preserves used_count
# for codes that already exist.
_seed_codes_raw = os.environ.get("SEED_ACCESS_CODES", "").strip()
if _seed_codes_raw:
    try:
        import json as _json
        _seed_list = _json.loads(_seed_codes_raw)
        _seed_conn = sqlite3.connect(ANALYTICS_DB)
        for _entry in _seed_list:
            _seed_conn.execute(
                "INSERT OR IGNORE INTO access_codes (code, label, expires_at, created_at, used_count) "
                "VALUES (?, ?, ?, ?, 0)",
                (_entry["code"], _entry.get("label", "Seeded"), _entry.get("expires_at"), datetime.utcnow().isoformat())
            )
            print(f"[startup] Access code ensured: {_entry['code']} ({_entry.get('label', '')})", flush=True)
        _seed_conn.commit()
        _seed_conn.close()
    except Exception as e:
        print(f"[startup] Seed access codes failed: {e}", flush=True)

# Legacy single-code seed — kept for backward compatibility
_seed_code = os.environ.get("SEED_ACCESS_CODE", "").strip()
if _seed_code:
    try:
        _seed_conn = sqlite3.connect(ANALYTICS_DB)
        _seed_conn.execute(
            "INSERT OR IGNORE INTO access_codes (code, label, expires_at, created_at, used_count) "
            "VALUES (?, 'Seed code', NULL, ?, 0)",
            (_seed_code, datetime.utcnow().isoformat())
        )
        _seed_conn.commit()
        _seed_conn.close()
        print(f"[startup] Seed access code ensured: {_seed_code}", flush=True)
    except Exception as e:
        print(f"[startup] Seed access code failed: {e}", flush=True)


@app.post("/analytics")
def log_analytics():
    try:
        body        = request.get_json(force=True, silent=True) or {}
        session_id  = str(body.get("session_id") or "")
        event       = str(body.get("event") or "")
        entity_type = body.get("entity_type") or None
        entity_id   = body.get("entity_id") or None
        entity_title = body.get("entity_title") or None
        meta        = body.get("meta") or None
        if not session_id or not event:
            return ("", 204)
        ts = datetime.utcnow().isoformat()
        conn = sqlite3.connect(ANALYTICS_DB)
        conn.execute(
            "INSERT INTO events (session_id, ts, event, entity_type, entity_id, entity_title, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, ts, event, entity_type, entity_id, entity_title, meta),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
    return ("", 204)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
