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
The courses are STEM-focused with some creative and health subjects.

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
  what could come after those?"\
"""

_WELCOME_INTERVIEW_SYSTEM = _FF_BASE_SYSTEM + """

## Your goal

Get enough information to make useful course suggestions. "Enough" is a low
bar — any signal that narrows the space is valuable. Examples of usable input:

- A subject area ("engineering", "something with computers", "creative work")
- A career name ("nurse", "electrician", "graphic designer")
- A preference about how they want to work ("with my hands", "outdoors",
  "with people", "with numbers")
- A constraint ("part-time", "apprenticeship route")
- An area they want to avoid ("not academic", "not sitting at a desk")

Any one of these is enough to pivot from interviewing to suggesting. Do not
hold out for richer input.
Courses can be filtered by the ssa_code field containing values (1 4 5 6 8 10 11 99)
which are SSA codes.

## How to behave

**Be warm, but stay on task.** Friendly in tone, but the goal is finding
courses, not building rapport for its own sake. Brief acknowledgement of what
the user says, then move forward. Don't probe feelings or family
circumstances. Don't ask how they're doing.

**Use Sonnet's judgement to interview intelligently.** When a user gives you
a partial signal, follow up on the most productive angle. If they say
"engineering", ask what kind of engineering appeals (mechanical, electrical,
civil, software). If they say "I like working with my hands", ask whether
they're drawn to fixing things, making things, or being outdoors. Triangulate
to narrow without making it feel like an interrogation.

**Keep your responses short.** 2-3 sentences typically. The user is on a
phone. Walls of text feel like a form.

**Stay within scope.** You exist to help find courses and careers at GMIoT.
If the user wants to talk about something else, gently redirect without
engaging on the off-topic content. If they disclose personal struggles or
sensitive content, acknowledge briefly and pivot back to course/career
territory — do not engage as a counsellor or friend.

**Respect the user's autonomy.** If they say "I don't know" twice, don't
push them through more questions. Offer alternatives. If they want to just
browse, let them.

## Escalation pattern

You have at most four interview turns before you should offer a graceful
exit. Each turn lowers the bar:

**Turn 1 (opening reply):** Acknowledge what the user said. If they gave
usable input, pivot immediately to suggesting courses (signal this in your
output — see below). If they gave nothing or were vague, ask a narrowing
question with two concrete starting points.

Example: "No worries — lots of people aren't sure at first. Two questions
that often help: are there subjects you enjoyed at school, even mildly? Or
jobs you've thought about, even briefly?"

**Turn 2 (if still vague):** Scaffold further with a different angle, often
negative elicitation.

Example: "That's fine. How about this: anything you'd rather *not* do? Sit
at a desk all day, work outside in bad weather, deal with the public, work
alone? Sometimes ruling things out is easier."

**Turn 3 (if still vague):** Offer the browse-everything escape hatch.

Example: "Want me to just show you some of the most popular courses at
GMIoT? You can browse what's on offer and see if anything catches your eye."

**Turn 4 (graceful exit, if user declines browse):** Suggest an advisor.

Example: "No worries at all. Sometimes it's easier to talk this through
with someone in person. GMIoT has advisors who are good at helping people
figure out where to start — you can [book a free course chat](https://gmiot.ac.uk/book-your-course-chat/) with one of their advisors. Would that be helpful?"

If at any point the user gives usable input, abandon the escalation and
pivot to suggesting courses. The escalation only progresses when the user
continues to give you nothing to work with.

## Safeguarding behaviour

If the user discloses sensitive content (mental health, family
difficulties, identity struggles, anything that suggests they need support
beyond course advice), acknowledge briefly with warmth and redirect to the
course/career frame. Do not probe. Do not validate at length. Do not give
advice on the disclosed topic.

Example: User says "I've been struggling with anxiety lately and I'm not
sure if I can handle uni."
Response: "That sounds tough to navigate. There are some good options to
consider — like part-time courses or apprenticeships that mix work with
study. Want to look at those? If it would help to talk it through with
someone first, you can [book a free course chat](https://gmiot.ac.uk/book-your-course-chat/) with a GMIoT advisor."

Brief acknowledgement, redirect to task, offer the advisor booking link
if it seems useful. Do not engage with the disclosure itself.

If the user asks about student support, advisors, or who to speak to,
always direct them to: [book a free course chat](https://gmiot.ac.uk/book-your-course-chat/)

## Pivoting to course suggestions

When you have usable input, your response should:

1. Briefly acknowledge what you've understood from the conversation.
2. Indicate you're going to show some relevant courses.
3. Signal this transition in your output by ending your message with the
   marker [PIVOT_TO_COURSES] on a new line.

The system will use this marker to trigger a course list response. Do not
list specific courses in your text — that happens through the system's
retrieval, not through your inference.

Example: "Got it — you're drawn to hands-on work and interested in
engineering specifically. Let me show you some courses that fit that.
[PIVOT_TO_COURSES]"

## Tone calibration examples

**Too cold (avoid):**
"Please specify your subject area."

**About right:**
"What kind of subjects are you drawn to?"

**Too warm (avoid):**
"I'd absolutely love to help you find the perfect course! Tell me all about
yourself — your dreams, your passions, what makes you tick!"

**Too personal (avoid):**
"How are you feeling about your future right now?"

**About right:**
"What sort of work would feel like you?"

## Filtering to a subject area

Only use [FILTER:N] when the user is asking for the whole of a top-level subject
area by its broad name — not for a specific discipline, role, or topic within it.

Use [FILTER:N] for requests like:
- "show me all engineering courses" → [FILTER:4]
- "what digital courses do you have?" → [FILTER:6]
- "I want to see construction options" → [FILTER:5]
- "show me health courses" → [FILTER:1]
- "what arts courses are there?" → [FILTER:10]
- "sport courses" → [FILTER:8]
- "social science" → [FILTER:11]
- "sustainability" → [FILTER:99]

Do NOT use [FILTER:N] for anything other than these exact subject areas.
In particular, never use [FILTER:N] for qualification types, levels, or providers —
those must use [PIVOT_TO_COURSES]. For example:
- "show me postgraduate courses" → [PIVOT_TO_COURSES] (not [FILTER:99])
- "what level 4 courses are there?" → [PIVOT_TO_COURSES]
- "show me apprenticeships" → [PIVOT_TO_COURSES]
- "courses at Salford" → [PIVOT_TO_COURSES]

Do NOT use [FILTER:N] for specific sub-disciplines or topics within an area —
those should also use [PIVOT_TO_COURSES] so the retrieval system can find the best
matches. For example:
- "show me electronics courses" → [PIVOT_TO_COURSES] (not [FILTER:4])
- "I want to do software development" → [PIVOT_TO_COURSES] (not [FILTER:6])
- "plumbing courses" → [PIVOT_TO_COURSES] (not [FILTER:5])

Use [FILTER:N] instead of [PIVOT_TO_COURSES] for these cases — do not use both.

Example: "Here are all the digital and technology courses at GMIoT. [FILTER:6]"

## Qualification pathway map

If the user asks about qualification types, levels, what different qualifications
mean, or how they relate to each other (e.g. "what's a T Level?", "what's the
difference between HNC and HND?", "what level should I be looking at?"), direct
them to the qualification pathway map by ending your response with [SHOW_QUAL_MAP].

Example: "Good question — there's a visual map that explains the different
qualification types and how they connect. Have a look and come back if you want
to explore courses from there. [SHOW_QUAL_MAP]"

Do not use [SHOW_QUAL_MAP] with [PIVOT_TO_COURSES] or [FILTER:N] in the same
response.

## Suggestion chips

When your response offers the user 2–4 concrete options to choose between,
append a [SUGGESTIONS:...] marker so the UI can render them as tappable chips.

Format: [SUGGESTIONS:option one|option two|option three]

Use this when you are giving the user specific things to pick from — narrow
sub-areas, work style preferences, or concrete examples. Keep each option
short (3–5 words). Do not use it for open-ended questions, for pivots to
courses, or for filter responses.

The sentence immediately before [SUGGESTIONS:...] must invite the user to
choose — it should read naturally as a lead-in to the options. Vary the
phrasing naturally.

Good examples:
- "Which of these sounds closest?"
- "Does any of these appeal?"
- "Which direction feels more like you?"
- "Pick whichever feels closest and we'll go from there."

Example response: "Great starting point. Creative industries covers a lot of
ground — are you drawn more to the technical side or the design side?
Which of these sounds closest?
[SUGGESTIONS:audio/visual production|game development|graphic design|digital media]"

Do not use [SUGGESTIONS:...] and [PIVOT_TO_COURSES] or [FILTER:N] in the
same response.

## What not to do

- Do not ask the user's name, age, or location.
- Do not ask about family or personal circumstances.
- Do not generate specific course names or details — those come from
  retrieval, not from your knowledge.
- Do not encourage personal disclosure ("tell me more about yourself").
- Do not promise outcomes ("this course will definitely lead to...").
- Do not use emojis.

## Post-pivot advisory mode

Once courses have been shown (you will be told in the dynamic note), the
interview phase is over. Be genuinely helpful with whatever the user asks
— draw on your knowledge to give real, useful answers. Do not narrow
yourself to only course-finding or deflect things you can answer.

**The advisor booking link is not a default deflection.** Use it only for
genuinely institution-specific questions you cannot answer — specific
application deadlines, whether a particular non-standard qualification is
accepted, bursary details. Do not use it as a substitute for advice you
can give yourself.

**Keep responses concise** — 3–4 sentences. Still mobile-readable.
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
        dynamic_note = "\n\n[Courses have been shown. You are now in advisory mode — see ## Post-pivot advisory mode in your instructions.]" + saved_note
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


def merge_candidates(list_a: list, list_b: list) -> list:
    """Merge two candidate lists, deduplicating by ID (highest score wins).

    Returns a list sorted by score descending, capped at TOP_N_CANDIDATES.
    """
    seen = {}
    for candidate in list_a + list_b:
        id_ = candidate["id"]
        if id_ not in seen or candidate["score"] > seen[id_]["score"]:
            seen[id_] = candidate
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:TOP_N_CANDIDATES]


# Currently used by legacy /chat. Retained — intended for reuse in welcome_chat
# filter extension (SQL-from-chat-text feature).
def build_where_clause(filters: dict, id_scope: list | None = None) -> dict:
    """Build a Chroma where clause for course searches.

    Always includes chunk=overview. Adds field filters from Haiku's spec.
    id_scope is a list of course_id strings for candidate-set scoping (Stage 3+).
    """
    conditions = [{"chunk": {"$eq": "overview"}}]
    if id_scope:
        conditions.append({"course_id": {"$in": [str(id_) for id_ in id_scope]}})
    if filters:
        if filters.get("ssa_label"):
            conditions.append({"ssa_label": {"$eq": filters["ssa_label"]}})
        if filters.get("qual_type"):
            # qual_type is now canonical — no expansion needed
            conditions.append({"qualification_type": {"$in": filters["qual_type"]}})
        if filters.get("mode"):
            conditions.append({"mode": {"$eq": filters["mode"]}})
        if filters.get("provider"):
            conditions.append({"provider": {"$eq": filters["provider"]}})
        if filters.get("level"):
            conditions.append({"level": {"$eq": filters["level"]}})
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def build_job_where_clause(id_scope: list | None = None) -> dict:
    """Build a Chroma where clause for job searches.

    Always includes chunk=overview. Optionally scopes to a list of job_id strings.
    """
    if not id_scope:
        return {"chunk": {"$eq": "overview"}}
    return {"$and": [
        {"chunk":  {"$eq": "overview"}},
        {"job_id": {"$in": [str(id_) for id_ in id_scope]}},
    ]}


def _fallback_spec(message: str) -> dict:
    """Minimal spec used when the specify_searches tool call fails."""
    return {
        "query_type":        "intent",
        "searches":          [{"query": message, "type": "both", "scope": "full_collection"}],
        "collection_action": "build",
        "acknowledgement":   "Here are some results for you.",
    }


def execute_specified_searches(
    spec: dict,
    candidate_set: dict | None = None,
) -> tuple[list, list]:
    """Execute the searches Haiku specified. Returns (job_candidates, course_candidates).

    Each candidate dict contains: type, id, title, score, full_text, _meta
    (plus qualification_type, level for courses).
    No score threshold — Top-N relative quality only.

    candidate_set: active set from session — used when search scope is "candidate_set".
    If scope is candidate_set but no set is active, falls back to full_collection.
    """
    all_job_hits    = []   # raw Chroma hit dicts, one per search that included jobs
    all_course_hits = []   # raw Chroma hit dicts, one per search that included courses

    for search in (spec.get("searches") or []):
        query_text  = search["query"]
        search_type = search["type"]          # "courses" | "jobs" | "both"
        filters     = search.get("filters") or {}
        scope       = search.get("scope", "full_collection")

        print(f"[execute_searches] embed: {query_text!r}", flush=True)
        vector = embed(query_text)

        # Resolve candidate-set scoping — fall back to full_collection if no active set
        course_id_scope = None
        job_id_scope    = None
        if scope == "candidate_set" and candidate_set:
            raw_cids = candidate_set.get("course_ids") or []
            raw_jids = candidate_set.get("job_ids")    or []
            # Chroma stores IDs as strings — keep as strings throughout
            course_id_scope = [str(id_) for id_ in raw_cids] or None
            job_id_scope    = [str(id_) for id_ in raw_jids] or None
            print(
                f"[scope] course_id_scope sample={course_id_scope[:3] if course_id_scope else []} "
                f"types={[type(v).__name__ for v in (course_id_scope or [])[:3]]}",
                flush=True,
            )

        course_where = build_where_clause(filters, id_scope=course_id_scope)
        job_where    = build_job_where_clause(id_scope=job_id_scope)

        # Structural filters (qual_type, ssa_label, etc.) do the selection — use a high
        # ceiling so every matching record is returned, not just the top-N by embedding.
        has_structural_filters = bool(
            filters.get("qual_type") or filters.get("ssa_label") or
            filters.get("mode")      or filters.get("provider")  or filters.get("level")
        )
        n_results = 200 if has_structural_filters else SEARCH_TOP_N

        if search_type in ("jobs", "both"):
            hits = jobs_col.query(
                query_embeddings=[vector],
                n_results=n_results,
                where=job_where,
                include=["metadatas", "distances", "documents"],
            )
            all_job_hits.append(hits)

        if search_type in ("courses", "both"):
            hits = match_courses_col.query(
                query_embeddings=[vector],
                n_results=n_results,
                include=["metadatas", "distances", "documents"],
            )
            all_course_hits.append(hits)

    # Batch-fetch skills chunks for all unique job IDs across all searches
    all_job_overview_ids = list(dict.fromkeys(
        id_
        for hits in all_job_hits
        for id_ in hits["ids"][0]
    ))
    skills_lookup = {}
    if all_job_overview_ids:
        skills_ids = [id_.replace("_overview", "_skills") for id_ in all_job_overview_ids]
        sk = jobs_col.get(ids=skills_ids, include=["documents"])
        for sk_id, sk_doc in zip(sk["ids"], sk["documents"]):
            skills_lookup[sk_id.replace("_skills", "")] = sk_doc

    # Build job candidates from all hits
    raw_job_candidates = []
    for hits in all_job_hits:
        for id_, meta, dist, ov_doc in zip(
            hits["ids"][0], hits["metadatas"][0],
            hits["distances"][0], hits["documents"][0],
        ):
            s       = score(dist)
            jid     = str(meta["job_id"])
            sk_text = skills_lookup.get(jid, "")
            print(f"[execute_searches] job: {meta.get('title')!r} score={s}", flush=True)
            raw_job_candidates.append({
                "type":      "job",
                "id":        jid,
                "title":     meta.get("title", ""),
                "score":     s,
                "full_text": ov_doc + "\n\n" + sk_text if sk_text else ov_doc,
                "_meta":     meta,
            })

    # Build course candidates from all hits
    raw_course_candidates = []
    for hits in all_course_hits:
        for cid, meta, dist, ov_doc in zip(
            hits["ids"][0], hits["metadatas"][0], hits["distances"][0], hits["documents"][0],
        ):
            s = score(dist)
            print(f"[execute_searches] course: {meta.get('title')!r} score={s}", flush=True)
            raw_course_candidates.append({
                "type":               "course",
                "id":                 cid,
                "title":              meta.get("title", ""),
                "score":              s,
                "full_text":          ov_doc,
                "_meta":              meta,
            })

    # Deduplicate by ID (highest score wins), sort descending, cap at TOP_N_CANDIDATES
    job_candidates    = merge_candidates(raw_job_candidates,    [])
    course_candidates = merge_candidates(raw_course_candidates, [])

    return job_candidates, course_candidates


def format_browsing_history(browsing_history: list) -> str:
    """Format browsing history as readable text for the Haiku system prompt."""
    if not browsing_history:
        return "None yet."
    return "\n".join(
        f"  {item.get('type', 'item').capitalize()}: {item.get('title', '')}"
        for item in browsing_history
    )


def augment_query_with_context(query: str, browsing_history: list) -> str:
    """Augment a short/ambiguous query with typed browsing history before embedding.

    Only augments when the query is under 6 words AND browsing history exists.
    Type-aware: career queries use only career titles, course queries use only
    course titles, ambiguous queries use the most recent 3 items of any type.
    Full specific queries (6+ words) are returned unchanged.
    """
    words = query.strip().split()
    if len(words) >= 6 or not browsing_history:
        return query

    career_keywords = {'career', 'careers', 'job', 'jobs', 'work', 'role', 'roles'}
    course_keywords = {'course', 'courses', 'study', 'qualification', 'learn'}
    query_lower = query.lower()

    wants_careers = any(k in query_lower for k in career_keywords)
    wants_courses = any(k in query_lower for k in course_keywords)

    if wants_careers:
        relevant = [i['title'] for i in browsing_history if i['type'] == 'career']
    elif wants_courses:
        relevant = [i['title'] for i in browsing_history if i['type'] == 'course']
    else:
        relevant = [i['title'] for i in browsing_history[-3:]]

    if not relevant:
        return query

    context_titles = " ".join(relevant[-3:])
    return f"{query} {context_titles}"


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


# LEGACY — used by /chat only (see comment above)
_SPECIFY_SEARCHES_TOOL = {
    "name": "specify_searches",
    "description": "Specify what searches to run to answer the user query. Called before any retrieval happens.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["filter", "intent", "refine", "swerve", "out_of_scope", "explain"],
                "description": (
                    "filter: structured category/field request. "
                    "intent: interest or goal expression. "
                    "refine: narrowing current candidate set. "
                    "swerve: domain change mid-session. "
                    "out_of_scope: unrelated to courses or careers. "
                    "explain: conceptual question about how qualifications work, what a level "
                    "means, how progression routes work, or what subject areas cover — needs a "
                    "direct answer, not a search. Only use this for genuinely conceptual questions. "
                    "Do NOT use explain when the user wants to see, find, or browse courses or jobs "
                    "(e.g. 'show me postgraduate courses', 'what courses are at level 7') — those "
                    "are filter or intent queries. Set searches to [] and collection_action to none."
                ),
            },
            "searches": {
                "type": "array",
                "description": "One or more searches to run. Use multiple searches to cover different angles on the same intent.",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Expanded search query — rephrase the user's words into terminology likely to appear in course and job descriptions",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["courses", "jobs", "both"],
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["candidate_set", "full_collection"],
                            "description": "candidate_set: search within active candidate set only. full_collection: search entire Chroma collection.",
                        },
                        "filters": {
                            "type": "object",
                            "description": "Optional structured field filters to apply alongside semantic search",
                            "properties": {
                                "ssa_label":  {"type": "string"},
                                "qual_type":  {"type": "array", "items": {"type": "string"}},
                                "mode":       {"type": "string", "enum": ["FT", "PT", "FT/PT"]},
                                "provider":   {"type": "string"},
                                "level":      {"type": "integer"},
                            },
                        },
                    },
                    "required": ["query", "type", "scope"],
                },
            },
            "collection_action": {
                "type": "string",
                "enum": ["build", "refine", "replace", "none"],
                "description": (
                    "build: create new candidate set from results. "
                    "refine: narrow existing set. "
                    "replace: discard existing set and build new one. "
                    "none: return focal card(s) only."
                ),
            },
            "acknowledgement": {
                "type": "string",
                "description": "One sentence for the bottom zone — what you understood and are doing",
            },
        },
        "required": ["query_type", "searches", "collection_action", "acknowledgement"],
    },
}

_SPECIFY_SEARCHES_SYSTEM = (
    "You are a search director for a course and career exploration app.\n"
    "When a user sends a query, you specify what searches to run before\n"
    "any results are fetched.\n\n"
    "Think carefully about what the user is asking:\n"
    '- "filter" queries name a category, subject, or field constraint\n'
    '  e.g. "show me construction courses", "part time health jobs"\n'
    '- "intent" queries express an interest, goal, or personal situation\n'
    '  e.g. "I want to work outdoors", "I\'m good with my hands"\n'
    '- "refine" queries narrow what the user is already looking at\n'
    '  e.g. "just the HND ones", "which of these are near Wigan"\n'
    '- "swerve" queries change domain mid-session\n'
    '  e.g. "actually show me digital courses instead"\n'
    "- \"out_of_scope\" queries are unrelated to courses or careers\n\n"
    "For search queries, expand the user's words into terminology likely\n"
    'to appear in course and job descriptions. "NHS" becomes "healthcare\n'
    'clinical nursing hospital medical". "Building industry" becomes\n'
    '"construction civil engineering built environment site management".\n\n'
    "You may specify multiple searches to cover different angles on the\n"
    "same intent. Each search can target courses, jobs, or both.\n\n"
    "If a candidate set is active, decide whether to search within it\n"
    "(refine) or the full collection (new search).\n\n"
    f"The app has these subject areas: {', '.join(label for label, _ in SUBJECT_AREAS)}.\n\n"
    f"Available providers: {', '.join(PROVIDERS.keys())}.\n\n"
    "For explain queries, set searches to [] and collection_action to none."
)

# LEGACY — used by /chat only (see comment above)
_SELECT_RESULTS_TOOL = {
    "name": "select_results",
    "description": "Select the most relevant results from the search results provided.",
    "input_schema": {
        "type": "object",
        "properties": {
            "approved_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of results to show the user",
            },
            "rejected_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of results to discard as irrelevant",
            },
            "acknowledgement": {
                "type": "string",
                "description": "One sentence for the bottom zone",
            },
            "advisory_trigger": {
                "type": "boolean",
                "description": "Whether to trigger an advisory card",
            },
        },
        "required": ["approved_ids", "acknowledgement", "advisory_trigger"],
    },
}


# LEGACY — used by /chat only (see comment above)
def chat_specify_searches(
    message: str,
    chat_history: list,
    browsing_history: list,
    candidate_set: dict | None = None,
) -> dict | None:
    """Call Haiku with the specify_searches tool. Returns the tool input dict, or None on failure."""
    history_str = format_browsing_history(browsing_history)
    prior_turns = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in (chat_history or [])[-6:]
    )
    context_block = ""
    if prior_turns:
        context_block += f"\n\nPrior conversation:\n{prior_turns}"
    if browsing_history:
        context_block += f"\n\nBrowsing history (oldest to newest):\n{history_str}"

    # Candidate set summary — tells Haiku whether a set is active and what it contains
    if candidate_set and (candidate_set.get("course_ids") or candidate_set.get("job_ids")):
        n_courses = len(candidate_set.get("course_ids") or [])
        n_jobs    = len(candidate_set.get("job_ids")    or [])
        parts = []
        if n_courses:
            parts.append(f"{n_courses} course{'s' if n_courses != 1 else ''}")
        if n_jobs:
            parts.append(f"{n_jobs} career{'s' if n_jobs != 1 else ''}")
        context_block += (
            f"\n\nActive candidate set: {', '.join(parts)}"
            f"\n(built from: {candidate_set.get('built_from', 'previous search')})"
        )
    else:
        context_block += "\n\nNo active candidate set."

    user_prompt = f'User query: "{message}"{context_block}'

    try:
        resp = _anthropic_post({
            "model":       HAIKU_MODEL,
            "max_tokens":  512,
            "system":      _SPECIFY_SEARCHES_SYSTEM,
            "tools":       [_SPECIFY_SEARCHES_TOOL],
            "tool_choice": {"type": "tool", "name": "specify_searches"},
            "messages":    [{"role": "user", "content": user_prompt}],
        }, call_site="specify_searches")
        data = resp.json()
        turn1_content = data["content"]
        spec = turn1_content[0]["input"]
        return spec, turn1_content, user_prompt
    except RateLimitError:
        return "rate_limited", None, None
    except Exception as e:
        print(f"[specify_searches] FAILED — {e}", flush=True)
        return None, None, None


def _log_specify_searches(spec: dict) -> None:
    """Log the specify_searches result to console in a readable format."""
    print("[SPECIFY_SEARCHES]", flush=True)
    print(f"  query_type: {spec.get('query_type')}", flush=True)
    print(f"  collection_action: {spec.get('collection_action')}", flush=True)
    searches = spec.get("searches", [])
    if searches:
        print("  searches:", flush=True)
        for s in searches:
            filters = s.get("filters")
            filters_str = f" filters={filters}" if filters else ""
            print(
                f"    - query: {s.get('query')!r}\n"
                f"      type: {s.get('type')}  scope: {s.get('scope')}{filters_str}",
                flush=True,
            )
    print(f"  acknowledgement: {spec.get('acknowledgement')!r}", flush=True)


def format_results_for_haiku(
    job_candidates: list,
    course_candidates: list,
    job_meta_by_id: dict,
    course_meta_by_id: dict,
    message: str,
    candidate_set: dict | None,
) -> str:
    """Format retrieved candidates as a concise summary string for Haiku turn 2."""
    lines = []
    if course_candidates:
        lines.append(f"Courses found ({len(course_candidates)}):")
        for c in course_candidates:
            meta = course_meta_by_id.get(c["id"]) or {}
            provider = meta.get("provider", "")
            level = c.get("level", "")
            level_str = f" · Level {level}" if level else ""
            lines.append(f"[{c['id']}] {c['title']} · {provider}{level_str}")
    if job_candidates:
        if lines:
            lines.append("")
        lines.append(f"Jobs found ({len(job_candidates)}):")
        for c in job_candidates:
            meta = job_meta_by_id.get(c["id"]) or {}
            source = meta.get("source", "").upper()
            lines.append(f"[{c['id']}] {c['title']} · {source}")
    lines.append(f'\nUser query: "{message}"')
    cs = candidate_set
    if cs and (cs.get("course_ids") or cs.get("job_ids")):
        lines.append(f"Candidate set active: Yes (built from: {cs.get('built_from', 'previous search')})")
    else:
        lines.append("Candidate set active: No")
    return "\n".join(lines)


_SELECT_RESULTS_SYSTEM = (
    "You have specified searches and the backend has fetched results. "
    "Review the results and select the ones that genuinely match what the user asked for. "
    "Reject results that are clearly off-topic or irrelevant to the user's query. "
    "Provide a brief acknowledgement (one sentence) describing what was found — "
    "say what you are showing the user, not what you searched for. "
    "Example: 'Found 5 health care courses across Wigan and Tameside.' "
    "or 'Showing 8 construction management courses from GM IoT partners.' "
    "When the selected results include career roles, end your acknowledgement with the sentence: "
    "'Tap any role to see where it could lead.' "
    "Do not include this sentence when results are courses only."
)


def chat_select_results(
    turn1_user_prompt: str,
    turn1_content: list,
    results_summary: str,
) -> dict | None:
    """Turn 2 — pass retrieved results back to Haiku for selection and acknowledgement.

    The Anthropic API requires a tool_result block immediately after a tool_use block.
    We pass the results summary as the tool_result content.
    """
    # Find the tool_use_id from turn 1 — required to close the tool_result loop
    tool_use_id = next(
        (b["id"] for b in turn1_content if b.get("type") == "tool_use"),
        None,
    )
    if not tool_use_id:
        print("[select_results] no tool_use_id in turn1_content — cannot build turn 2", flush=True)
        return None

    turn2_user_content = [
        {
            "type":        "tool_result",
            "tool_use_id": tool_use_id,
            "content":     (
                f"Searches complete. Here are the results:\n{results_summary}\n\n"
                "Select the most relevant results and provide your acknowledgement."
            ),
        }
    ]
    try:
        resp = _anthropic_post({
            "model":       HAIKU_MODEL,
            "max_tokens":  512,
            "system":      _SELECT_RESULTS_SYSTEM,
            "tools":       [_SELECT_RESULTS_TOOL],
            "tool_choice": {"type": "tool", "name": "select_results"},
            "messages": [
                {"role": "user",      "content": turn1_user_prompt},
                {"role": "assistant", "content": turn1_content},
                {"role": "user",      "content": turn2_user_content},
            ],
        }, call_site="select_results")
        selection = resp.json()["content"][0]["input"]
        return selection
    except RateLimitError:
        return None
    except Exception as e:
        print(f"[select_results] FAILED — {e}", flush=True)
        return None


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


# LEGACY — /chat is not called by the current frontend (futurefinder-chat-first).
# All chat goes through /chat/welcome. This endpoint and its supporting functions
# (chat_specify_searches, chat_select_results, _SPECIFY_SEARCHES_TOOL, etc.) are
# retained as reference for the planned welcome_chat filter extension.
@app.post("/chat")
def chat():
    cleanup_sessions()
    body             = request.get_json(force=True)
    message          = (body.get("message") or "").strip()
    context          = body.get("session_context") or []
    chat_history     = body.get("chat_history") or []
    browsing_history = body.get("browsing_history") or []
    saved_items      = body.get("saved_items") or {"courses": [], "careers": []}
    candidate_set    = body.get("candidate_set") or None
    session_id       = body.get("session_id") or "default"

    if not message:
        return jsonify({"error": "message is required"}), 400

    get_session(session_id)  # ensure session exists before logging count
    print(f"[session] {session_id[:8]}... active sessions: {len(_sessions)}", flush=True)
    print(f"[chat] message={message!r}", flush=True)
    print(f"[chat] session_context count={len(context)}", flush=True)
    print(f"[chat] chat_history turns={len(chat_history)}", flush=True)
    print(f"[chat] browsing_history count={len(browsing_history)}", flush=True)

    # Stages 1–3 — Haiku specifies searches before any retrieval fires
    spec, turn1_content, turn1_user_prompt = chat_specify_searches(message, chat_history, browsing_history, candidate_set)
    if spec == "rate_limited":
        return jsonify({
            "results":         [],
            "acknowledgement": "The service is very busy right now — please try again in a moment.",
            "search_type":     "none",
            "candidate_set":   candidate_set,
        })
    if spec:
        _log_specify_searches(spec)
    else:
        print("[SPECIFY_SEARCHES] no result (tool call failed) — using fallback", flush=True)
        spec = _fallback_spec(message)
        turn1_content = None
        turn1_user_prompt = None

    # Stage 2 — out_of_scope: short-circuit before retrieval (candidate set unchanged)
    if spec.get("query_type") == "out_of_scope":
        return jsonify({
            "results":         [],
            "acknowledgement": spec.get("acknowledgement") or "I can only help with courses and careers — try asking about a subject area or job role.",
            "search_type":     "none",
            "candidate_set":   candidate_set,
        })

    # Stage 2 — explain: answer directly without searching
    if spec.get("query_type") == "explain":
        answer = chat_explain(message, chat_history)
        print(f"[chat] explain response: {answer[:80]!r}", flush=True)
        return jsonify({
            "results":         [],
            "response_text":   answer,
            "acknowledgement": spec.get("acknowledgement") or "Here's how that works.",
            "search_type":     "none",
            "candidate_set":   candidate_set,
        })

    # Stage 2 — execute Haiku's specified searches, scoped to candidate set when requested
    job_candidates, course_candidates = execute_specified_searches(spec, candidate_set)

    job_ft_by_id       = {c["id"]: c["full_text"] for c in job_candidates}
    job_meta_by_id     = {c["id"]: c.pop("_meta") for c in job_candidates}
    job_score_by_id    = {c["id"]: c["score"]     for c in job_candidates}
    course_meta_by_id  = {c["id"]: c.pop("_meta") for c in course_candidates}
    course_score_by_id = {c["id"]: c["score"]     for c in course_candidates}

    # Stage 3 — update candidate set according to collection_action
    # refine is disabled — treated as none until Stage 4 gatekeeping is in place.
    # None/missing collection_action defaults to none.
    collection_action = spec.get("collection_action") or "none"

    if collection_action in ("build", "replace"):
        new_candidate_set = {
            "course_ids": [c["id"] for c in course_candidates],
            "job_ids":    [c["id"] for c in job_candidates],
            "built_from": f'Chat: "{message[:50]}"',
        }
    else:
        # "none", "refine" (disabled), or any unrecognised value — pass set through unchanged
        new_candidate_set = candidate_set

    print(
        f"[chat] collection_action={collection_action!r} "
        f"new_set courses={len((new_candidate_set or {}).get('course_ids') or [])} "
        f"jobs={len((new_candidate_set or {}).get('job_ids') or [])}",
        flush=True,
    )

    # Count this as a qualifying interaction
    _increment_qualifying(session_id)

    if not job_candidates and not course_candidates:
        return jsonify({
            "results":         [],
            "acknowledgement": "I couldn't find anything matching that — try a different search term.",
            "search_type":     "none",
            "candidate_set":   new_candidate_set,
        })

    # Stage 4 — two-turn Haiku gatekeeping
    if turn1_content and turn1_user_prompt:
        results_summary = format_results_for_haiku(
            job_candidates, course_candidates,
            job_meta_by_id, course_meta_by_id,
            message, candidate_set,
        )
        selection = chat_select_results(turn1_user_prompt, turn1_content, results_summary)
        if selection:
            approved_ids = {str(i) for i in (selection.get("approved_ids") or [])}
            rejected_ids = {str(i) for i in (selection.get("rejected_ids") or [])}
            print(
                f"[select_results] approved={sorted(approved_ids)} "
                f"rejected={sorted(rejected_ids)}",
                flush=True,
            )
            print(f"[select_results] ack={selection.get('acknowledgement')!r}", flush=True)
            job_candidates    = [c for c in job_candidates    if c["id"] in approved_ids]
            course_candidates = [c for c in course_candidates if c["id"] in approved_ids]
            # Refresh candidate set to contain only approved items
            if collection_action in ("build", "replace"):
                new_candidate_set = {
                    "course_ids": [c["id"] for c in course_candidates],
                    "job_ids":    [c["id"] for c in job_candidates],
                    "built_from": new_candidate_set.get("built_from", f'Chat: "{message[:50]}"'),
                }
            ack = selection.get("acknowledgement") or spec.get("acknowledgement") or "Here are some results for you."
        else:
            print("[select_results] FAILED — falling back to all turn 1 results", flush=True)
            ack = spec.get("acknowledgement") or "Here are some results for you."
    else:
        ack = spec.get("acknowledgement") or "Here are some results for you."

    # SQLite fetch for all candidates
    results = []

    for c in job_candidates:
        jid  = c["id"]
        meta = job_meta_by_id.get(jid)
        if not meta:
            continue
        db  = job_row(jid)
        job = format_job(meta, db, job_score_by_id[jid])
        job["full_text"] = job_ft_by_id[jid]
        results.append(job)

    for c in course_candidates:
        cid = c["id"]
        db  = ff_course_row(cid)
        if not db:
            continue
        results.append(format_course_from_db(db, course_score_by_id.get(cid, 0)))

    if not results:
        ack = "I couldn't find anything relevant to that — try a different topic."

    print(f"[chat] returning {len(results)} results", flush=True)

    # Advisory check — only when results are present (meaningful exploration)
    advisory = check_advisory(context, session_id) if results else None

    response = {
        "results":         results,
        "acknowledgement": ack,
        "search_type":     "both",
        "candidate_set":   new_candidate_set,
    }
    if advisory:
        response["advisory"] = advisory
        print(f"[chat] advisory attached: {advisory['type']} id={advisory['id']}", flush=True)

    return jsonify(response)


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

# Seed a permanent access code from environment if set
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
