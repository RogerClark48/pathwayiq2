import json
import os
import sqlite3
import time
from datetime import datetime
import httpx
import numpy as np
import voyageai
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from threading import Lock
import chromadb

load_dotenv()
print(f"[startup] VOYAGE_API_KEY present: {bool(os.environ.get('VOYAGE_API_KEY'))}", flush=True)
print(f"[startup] All env vars: {[k for k in os.environ.keys()]}", flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BASE              = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH        = os.path.join(_BASE, "chroma_store")
COURSES_DB         = os.path.join(_BASE, "emiot.sqlite")   # v1 only — dead in v2, do not use
GMIOT_DB           = os.path.join(_BASE, "gmiot.sqlite")
JOBS_DB            = os.path.join(_BASE, "job_roles_asset.db")
CONNECTIONS_DB     = os.path.join(_BASE, "connections.db")
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

# Maps subject tile labels to exact ssa_label values in gmiot_courses
SSA_MAP = {
    'Engineering':   'Engineering and Manufacturing Technologies',
    'Digital & Tech':'Information and Communication Technology',
    'Construction':  'Construction, Planning and the Built Environment',
    'Health':        'Health, Public Services and Care',
    'Arts & Media':  'Arts, Media and Publishing',
}

# Maps qual tile filter labels to the qual_type values they cover in gmiot_courses
QUAL_FILTER_MAP = {
    'T Level':          ['T Level'],
    'Apprenticeship':   ['Apprenticeship'],
    'HNC':              ['HNC', 'HNC/HTQ', 'HTQ', 'HNC/HND'],
    'HND':              ['HND', 'HND/HTQ', 'HNC/HND'],
    'Foundation Degree':['FdA', 'FdSc', 'CertHE', 'DipHE'],
    "Bachelor's Degree":['BA Hons', 'BEng Hons', 'BSc Hons'],
    "Master's Degree":  ['MSc'],
    'Access to HE':     ['Access to HE'],
    'Short Course':     ['Award', 'Short Course'],
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
courses_col          = chroma.get_collection("gmiot_courses")
jobs_col             = chroma.get_collection("gmiot_jobs")
courses_learning_col = chroma.get_collection("gmiot_courses_learning")
jobs_skills_col      = chroma.get_collection("gmiot_jobs_skills")

CAUTION_DIVERGENCE_THRESHOLD  = 15  # domain% - skills% > this → caution flag
CROSS_COLLECTION_MIN_SKILLS   = 72  # hard floor — connections below this are excluded
CROSS_COLLECTION_MIN_DOMAIN   = 75  # hard floor — low domain score connections excluded

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


def course_row(course_id: str) -> dict | None:
    conn = sqlite3.connect(COURSES_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM Course WHERE courseId = ?", (course_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def gmiot_course_row(course_id: str) -> dict | None:
    conn = sqlite3.connect(GMIOT_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM gmiot_courses WHERE course_id = ?", (course_id,)
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


def format_course_from_db(db: dict, match_score: int) -> dict:
    """Build a course result dict from a gmiot_courses SQLite row (no Chroma metadata)."""
    return {
        "type":               "course",
        "id":                 str(db["course_id"]),
        "title":              db["course_title"],
        "provider":           db["provider"],
        "subject_area":       db.get("subject_area"),
        "level":              db.get("level"),
        "qualification_type": db.get("qual_type"),
        "ssa_category":       db.get("ssa_label"),
        "source_url":         db.get("course_url"),
        "match_score":        match_score,
        "overview":           (db.get("overview") or "")[:500],
    }


def keyword_course_search(q: str, qualification: str | None) -> list[dict]:
    """SQLite LIKE search on course_title in gmiot_courses. Returns exact-title matches first.
    Only includes courses that have been embedded in Chroma (have an overview chunk)."""
    conn = sqlite3.connect(GMIOT_DB)
    conn.row_factory = sqlite3.Row
    sql    = "SELECT * FROM gmiot_courses WHERE course_title LIKE ?"
    params: list = [f"%{q}%"]
    if qualification:
        qual_values = QUAL_FILTER_MAP.get(qualification, [qualification])
        placeholders = ",".join("?" * len(qual_values))
        sql += f" AND qual_type IN ({placeholders})"
        params.extend(qual_values)
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        return []

    # Filter to courses that have Chroma embeddings
    candidate_ids = [str(dict(row)["course_id"]) for row in rows]
    chroma_ids    = [f"{cid}_overview" for cid in candidate_ids]
    stored        = courses_col.get(ids=chroma_ids, include=[])
    embedded      = {sid.replace("_overview", "") for sid in stored["ids"]}

    q_lower = q.lower()
    results = []
    for row in rows:
        db = dict(row)
        if str(db["course_id"]) not in embedded:
            continue
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


LEVEL_REF = """Qualification level reference:
T Level               -> Level 3
Higher Apprenticeship -> Level 4-5
HNC                   -> Level 4
HND                   -> Level 5
Bachelor's Degree     -> Level 6
Master's Degree       -> Level 7"""

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GATEKEEP_MODEL  = "llama3.2:3b"


def _parse_llm_json(content: str) -> dict:
    """Strip optional markdown fences and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    return json.loads(content)


def gatekeep_jobs(course_name, course_level, course_subject,
                  qualification_type, candidates):
    """Filter job candidates by subject/level relevance against a course.

    candidates: list of dicts with keys job_id, title, full_text, score
    Returns: (approved_ids, rejected_ids, acknowledgement)
    """
    lines = []
    for c in candidates:
        lines.append(
            f"JOB_ID={c['job_id']} | {c['title']} ({c['score']}%)\n"
            f"{c['full_text'][:600]}"
        )

    system_prompt = (
        "You are a careers advisor assistant. Review candidate career matches "
        "for a course and filter out any that are inappropriate by subject domain "
        "or qualification level.\n\n"
        "Respond with valid JSON only. No preamble, no explanation outside the JSON, "
        "no markdown fences. Return exactly:\n"
        '{"approved_ids": ["971", "846"], "rejected_ids": ["203"], '
        '"acknowledgement": "One sentence for the user."}\n\n'
        "Rules:\n"
        "- approved_ids and rejected_ids must contain the exact JOB_ID numbers "
        "from the candidates — do not return sequence numbers or any other value\n"
        "- approved_ids: job IDs genuinely relevant to the course subject AND "
        "appropriate entry level for the qualification\n"
        "- rejected_ids: job IDs in a different subject domain OR clearly mismatched level\n"
        "- acknowledgement: one short sentence the user will read — do not mention "
        "filtering or rejection, just frame results positively\n"
        "- When in doubt, approve — only reject clear mismatches\n"
        "- Never invent facts about courses or careers"
    )

    user_prompt = (
        f"Course: {course_name}\n"
        f"Qualification: {qualification_type} (Level {course_level})\n"
        f"Subject area: {course_subject}\n\n"
        f"{LEVEL_REF}\n\n"
        f"Candidate career matches to review:\n\n"
        + "\n\n---\n\n".join(lines)
        + "\n\nReturn JSON only."
    )

    try:
        resp = httpx.post(
            OLLAMA_CHAT_URL,
            json={
                "model": GATEKEEP_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        parsed = _parse_llm_json(resp.json()["message"]["content"])
        approved = [str(x) for x in parsed.get("approved_ids", [])]
        rejected = [str(x) for x in parsed.get("rejected_ids", [])]
        ack      = parsed.get("acknowledgement", "")
        print(f"[gatekeep_jobs] approved={approved} rejected={rejected}", flush=True)
        return approved, rejected, ack
    except Exception as e:
        print(f"[gatekeep_jobs] LLM failed ({e}) — approving all", flush=True)
        return [str(c["job_id"]) for c in candidates], [], ""


def gatekeep_courses(job_title, job_skills_text, candidates):
    """Filter course candidates by subject/level relevance against a job.

    candidates: list of dicts with keys course_id, title, level,
                qualification_type, ssa_category, overview, score
    Returns: (approved_ids, rejected_ids, acknowledgement)
    """
    lines = []
    for c in candidates:
        lines.append(
            f"COURSE_ID={c['course_id']} | {c['title']} — "
            f"{c['qualification_type']} Level {c['level']} ({c['score']}%)\n"
            f"{c['overview'][:400]}"
        )

    system_prompt = (
        "You are a careers advisor assistant. Review candidate course matches "
        "for a job and filter out any that are inappropriate by subject domain "
        "or qualification level.\n\n"
        "Respond with valid JSON only. No preamble, no explanation outside the JSON, "
        "no markdown fences. Return exactly:\n"
        '{"approved_ids": ["14", "3"], "rejected_ids": ["7"], '
        '"acknowledgement": "One sentence for the user."}\n\n'
        "Rules:\n"
        "- approved_ids and rejected_ids must contain the exact COURSE_ID numbers "
        "from the candidates — do not return sequence numbers or any other value\n"
        "- approved_ids: course IDs genuinely relevant to the job subject AND "
        "appropriate qualification level for the job's entry requirements\n"
        "- rejected_ids: course IDs in a different subject domain OR clearly mismatched level\n"
        "- acknowledgement: one short sentence the user will read — do not mention "
        "filtering or rejection, just frame results positively\n"
        "- When in doubt, approve — only reject clear mismatches\n"
        "- Never invent facts about courses or careers"
    )

    user_prompt = (
        f"Job role: {job_title}\n"
        f"Job entry requirements and skills:\n{job_skills_text[:600]}\n\n"
        f"{LEVEL_REF}\n\n"
        f"Candidate courses to review:\n\n"
        + "\n\n---\n\n".join(lines)
        + "\n\nReturn JSON only."
    )

    try:
        resp = httpx.post(
            OLLAMA_CHAT_URL,
            json={
                "model": GATEKEEP_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        parsed = _parse_llm_json(resp.json()["message"]["content"])
        approved = [str(x) for x in parsed.get("approved_ids", [])]
        rejected = [str(x) for x in parsed.get("rejected_ids", [])]
        ack      = parsed.get("acknowledgement", "")
        print(f"[gatekeep_courses] approved={approved} rejected={rejected}", flush=True)
        return approved, rejected, ack
    except Exception as e:
        print(f"[gatekeep_courses] LLM failed ({e}) — approving all", flush=True)
        return [str(c["course_id"]) for c in candidates], [], ""


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
            # Expand label-level names ("HND") to all matching qual_type values
            # ("HND", "HND/HTQ", "HNC/HND") — same mapping the tile path uses
            expanded = []
            for qt in filters["qual_type"]:
                expanded.extend(QUAL_FILTER_MAP.get(qt, [qt]))
            conditions.append({"qualification_type": {"$in": expanded}})
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
            hits = courses_col.query(
                query_embeddings=[vector],
                n_results=n_results,
                where=course_where,
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
        for meta, dist, ov_doc in zip(
            hits["metadatas"][0], hits["distances"][0], hits["documents"][0],
        ):
            s   = score(dist)
            cid = str(meta["course_id"])
            print(f"[execute_searches] course: {meta.get('course_name')!r} score={s}", flush=True)
            raw_course_candidates.append({
                "type":               "course",
                "id":                 cid,
                "title":              meta.get("course_name", ""),
                "score":              s,
                "qualification_type": meta.get("qualification_type", ""),
                "level":              meta.get("level", ""),
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


_EXPLAIN_SYSTEM = (
    "You are a course and career guidance advisor for GMIoT — Greater Manchester's "
    "Institute of Technology. A student has asked a question about qualifications, "
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

    "PARTNER PROVIDERS (all in Greater Manchester):\n"
    "Wigan & Leigh College — Wigan\n"
    "University of Salford — Salford\n"
    "Trafford & Stockport College — campuses in Stretford and Stockport\n"
    "Tameside College — Ashton-under-Lyne\n"
    "Bury College — Bury\n"
    "Ada College — Manchester city centre; specialises in digital and technology\n\n"

    "SUBJECT AREAS COVERED:\n"
    "Engineering and Manufacturing Technologies — mechanical, electrical, "
    "manufacturing, automotive\n"
    "Information and Communication Technology — software development, networking, "
    "cybersecurity, data\n"
    "Construction, Planning and the Built Environment — building, civil engineering, "
    "architecture, surveying\n"
    "Health, Public Services and Care — nursing, healthcare, social care\n"
    "Arts, Media and Publishing — creative arts, graphic design, media production\n"
    "Business, Administration and Law — business management, finance, administration\n\n"

    "JOB DATA SOURCES:\n"
    "NCS — National Careers Service; UK government careers information\n"
    "Prospects — UK graduate careers website with detailed job role information\n\n"

    "Do not invent course titles, job titles, or facts not grounded in the above. "
    "If you genuinely do not know, say so briefly and suggest the user explore the app. "
    "Do not use markdown formatting — no bold, no bullet points, plain text only."
)


def chat_explain(message: str, chat_history: list) -> str:
    """Direct Haiku call to answer a qualifications/pathway question. No tool use, no search.

    Returns plain text answer, or a fallback string on failure.
    """
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]
    messages = [{"role": m["role"], "content": m["content"]} for m in chat_history]
    messages.append({"role": "user", "content": message})
    try:
        resp = httpx.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      HAIKU_MODEL,
                "max_tokens": 300,
                "system":     _EXPLAIN_SYSTEM,
                "messages":   messages,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    except Exception as e:
        print(f"[chat_explain] FAILED — {e}", flush=True)
        return "I'm not able to answer that right now — try exploring the subject areas or qualifications above."


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
                    "explain: question about how qualifications work, what levels mean, "
                    "progression routes, providers, or subject areas — needs a direct answer, "
                    "not a search. Set searches to [] and collection_action to none."
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
    "The app has these subject areas: Engineering and Manufacturing\n"
    "Technologies, Information and Communication Technology, Construction\n"
    "Planning and the Built Environment, Health Public Services and Care,\n"
    "Arts Media and Publishing, Business Administration and Law.\n\n"
    "Available providers: Wigan & Leigh College, University of Salford,\n"
    "Trafford & Stockport College, Tameside College, Bury College,\n"
    "Ada College.\n\n"
    "For explain queries, set searches to [] and collection_action to none."
)

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
        resp = httpx.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":       HAIKU_MODEL,
                "max_tokens":  512,
                "system":      _SPECIFY_SEARCHES_SYSTEM,
                "tools":       [_SPECIFY_SEARCHES_TOOL],
                "tool_choice": {"type": "tool", "name": "specify_searches"},
                "messages":    [{"role": "user", "content": user_prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        turn1_content = data["content"]
        spec = turn1_content[0]["input"]
        return spec, turn1_content, user_prompt
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
        resp = httpx.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
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
            },
            timeout=30,
        )
        resp.raise_for_status()
        selection = resp.json()["content"][0]["input"]
        return selection
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
        course_hits = courses_col.query(
            query_embeddings=[vector],
            n_results=10,
            where={"chunk": {"$eq": "overview"}},
            include=["metadatas", "distances", "documents"],
        )
        for meta, dist, doc in zip(
            course_hits["metadatas"][0], course_hits["distances"][0], course_hits["documents"][0],
        ):
            cid = str(meta["course_id"])
            if cid in seen_set:
                continue
            candidates.append({
                "type":               "course",
                "id":                 cid,
                "title":              meta.get("course_name", ""),
                "score":              score(dist),
                "qualification_type": meta.get("qualification_type", ""),
                "level":              meta.get("level", ""),
                "full_text":          doc[:300],
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
        resp = httpx.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":       SONNET_MODEL,
                "max_tokens":  300,
                "system":      system_prompt,
                "tools":       [_ADVISORY_TOOL],
                "tool_choice": {"type": "tool", "name": "submit_advisory_decision"},
                "messages":    [{"role": "user", "content": user_prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        result      = resp.json()["content"][0]["input"]
        item_type   = result.get("advisory_item_type", "none")
        item_id     = (result.get("advisory_item_id") or "").strip()
        explanation = result.get("advisory_explanation", "")
        trigger     = result.get("advisory_trigger", "")
        print(f"[advisory_llm] trigger={trigger!r} type={item_type} id={item_id!r}", flush=True)

        if item_type == "none" or not item_id:
            return None
        return {"type": item_type, "id": item_id, "explanation": explanation}
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
        db = gmiot_course_row(advisory["id"])
        if not db:
            return None
        advisory["title"]              = db["course_title"]
        advisory["provider"]           = db["provider"]
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
        resp = httpx.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":       HAIKU_MODEL,
                "max_tokens":  1000,
                "system":      system_prompt,
                "tools":       [_CHAT_TOOL],
                "tool_choice": {"type": "tool", "name": "submit_chat_result"},
                "messages":    chat_history + [{"role": "user", "content": user_prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        result           = resp.json()["content"][0]["input"]
        approved_jobs    = [str(x) for x in result.get("approved_job_ids", [])]
        approved_courses = [str(x) for x in result.get("approved_course_ids", [])]
        ack              = result.get("acknowledgement", "")
        is_off_topic     = bool(result.get("is_off_topic", False))
        print(f"[chat_llm] tool call received — approved_jobs={approved_jobs} approved_courses={approved_courses}", flush=True)
        return approved_jobs, approved_courses, ack, is_off_topic
    except Exception as e:
        print(f"[chat_llm] API call failed ({e}) — approving all", flush=True)
        approved_jobs    = [c["id"] for c in candidates if c["type"] == "job"]
        approved_courses = [c["id"] for c in candidates if c["type"] == "course"]
        return approved_jobs, approved_courses, "Here are some results for you.", False


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

@app.get("/search/courses")
def search_courses():
    subject       = request.args.get("subject", "").strip()
    q             = request.args.get("q", "").strip()
    qualification = request.args.get("qualification", "").strip()

    # Subject-tile path — direct SQLite lookup by SSA label, no embedding, no limit
    if subject:
        ssa_label = SSA_MAP.get(subject)
        if not ssa_label:
            return jsonify({"subject": subject, "results": [], "message": "Unknown subject area."})

        conn = sqlite3.connect(GMIOT_DB)
        conn.row_factory = sqlite3.Row
        sql    = "SELECT * FROM gmiot_courses WHERE ssa_label = ? "
        params: list = [ssa_label]
        if qualification:
            qual_values  = QUAL_FILTER_MAP.get(qualification, [qualification])
            placeholders = ",".join("?" * len(qual_values))
            sql += f"AND qual_type IN ({placeholders}) "
            params.extend(qual_values)
        sql += "ORDER BY course_title"
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

    where_clause = {"chunk": {"$eq": "overview"}}
    if qualification:
        where_clause = {"$and": [
            {"chunk":              {"$eq": "overview"}},
            {"qualification_type": {"$contains": qualification}},
        ]}

    hits = courses_col.query(
        query_embeddings=[vector],
        n_results=100,
        where=where_clause,
        include=["metadatas", "distances"],
    )

    vector_results = []
    for meta, dist in zip(hits["metadatas"][0], hits["distances"][0]):
        s = score(dist)
        if s >= MIN_SCORE:
            db = gmiot_course_row(meta["course_id"])
            vector_results.append(format_course(meta, db, s))

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
                stored_meta = courses_col.get(
                    ids=[f"{course_id}_overview"], include=["metadatas"]
                )
                course_name = (
                    stored_meta["metadatas"][0].get("course_name")
                    if stored_meta["ids"] else None
                )

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
    # Lift the overview chunk vector — matches against job duties (overview chunks)
    stored = courses_col.get(
        ids=[f"{course_id}_overview"],
        include=["embeddings", "metadatas"],
    )
    if not stored["ids"]:
        # Fallback to skills chunk if no overview chunk
        stored = courses_col.get(
            ids=[f"{course_id}_skills"],
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
    limit = min(int(request.args.get("limit", 3)), 20)

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

    hits = courses_col.query(
        query_embeddings=[vector],
        n_results=limit,
        where={"chunk": {"$eq": "overview"}},
        include=["metadatas", "distances"],
    )

    results = []
    for meta, dist in zip(hits["metadatas"][0], hits["distances"][0]):
        s = score(dist)
        if s >= MIN_SCORE:
            cid          = str(meta["course_id"])
            db           = gmiot_course_row(cid)
            course       = format_course(meta, db, s)

            if s < CROSS_COLLECTION_MIN_DOMAIN:
                print(f"[caution] course {cid} -> job {job_id}: domain={s}% EXCLUDED (domain below floor)", flush=True)
                continue

            skills_score = compute_skills_score(cid, job_id)
            sk_pct       = f"{skills_score}%" if skills_score is not None else "N/A"

            if skills_score is None or skills_score < CROSS_COLLECTION_MIN_SKILLS:
                print(f"[caution] course {cid} -> job {job_id}: domain={s}% skills={sk_pct} EXCLUDED (skills below floor)", flush=True)
                continue

            caution = (s - skills_score) > CAUTION_DIVERGENCE_THRESHOLD if skills_score is not None else False
            course["skills_score"] = skills_score
            course["caution"]      = caution

            flag = "FLAGGED" if caution else "ok"
            print(f"[caution] course {cid} -> job {job_id}: domain={s}% skills={sk_pct} d={s - (skills_score or s):+d}% {flag}", flush=True)

            results.append(course)

    return jsonify({
        "job_id":    job_id,
        "job_title": job_meta.get("title"),
        "results":   results,
    })


@app.get("/courses/<int:course_id>")
def course_detail(course_id):
    db = gmiot_course_row(str(course_id))
    if not db:
        return jsonify({"error": f"Course {course_id} not found"}), 404
    return jsonify({
        "id":                   db["course_id"],
        "title":                db["course_title"],
        "provider":             db["provider"],
        "subject_area":         db.get("subject_area"),
        "level":                db.get("level"),
        "qual_type":            db.get("qual_type"),
        "mode":                 db.get("mode"),
        "course_url":           db.get("course_url"),
        "ssa_code":             db.get("ssa_code"),
        "ssa_label":            db.get("ssa_label"),
        "overview":             db.get("overview") or "",
        "what_you_will_learn":  db.get("what_you_will_learn") or "",
        "entry_requirements":   db.get("entry_requirements") or "",
        "progression":          db.get("progression") or "",
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
    })


@app.get("/jobs/<int:job_id>/progression")
def job_progression(job_id):
    jobs_conn = sqlite3.connect(JOBS_DB)
    jobs_conn.row_factory = sqlite3.Row

    # Step 1 — Check cache
    cached = jobs_conn.execute(
        "SELECT narrative, inbound_json, outbound_json FROM job_progression_cache "
        "WHERE job_id = ? AND prompt_version = 4", (job_id,)
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
        f"AUTHORITATIVE ENTRY ROUTES (written by career experts — treat as definitive for how "
        f"people reach this role and what qualifications or experience are typically required):\n"
        f"{job['entry_routes']}\n\n"
        f"AUTHORITATIVE CAREER PROGRESSION (written by career experts — treat as definitive for "
        f"where this role leads and what the natural next steps are):\n"
        f"{job['career_prospects'] or job['progression']}\n\n"
        f"Here are {len(candidates)} candidate job profiles from our database:\n\n"
        f"{candidate_block}\n\n"
        f"Your task:\n"
        f"1. Identify up to 4 candidates that someone might typically come FROM before reaching "
        f"this role — roles that naturally lead here, usually at a lower seniority level. "
        f"Use the authoritative entry routes above to guide your selection. "
        f"Only include roles that are a genuinely close fit. Fewer strong connections are better than "
        f"padding the list with weak ones. If this is an entry-level role, there may be no natural "
        f"preceding roles — return an empty inbound array rather than forcing connections.\n"
        f"2. Identify up to 4 candidates this role might naturally progress TO — roles at a "
        f"higher seniority or broader responsibility level. "
        f"Use the authoritative career progression above to guide your selection. "
        f"Only include roles that are a genuinely close fit. Fewer strong connections are better than "
        f"padding the list with weak ones. If this is a senior or specialist role near the top of its "
        f"field, there may be no natural outbound roles — return an empty outbound array rather than "
        f"forcing connections.\n"
        f"3. Write 2–3 sentences of warm, plain-English guidance explaining the progression "
        f"landscape for this role, suitable for a college student considering their future career. "
        f"Draw on the specific routes, qualifications, and next steps described in the authoritative "
        f"fields above — use their language and detail to make the narrative specific and grounded. "
        f"Keep it practical and directly relevant to this role.\n\n"
        f"Only select candidates from the list provided. If no candidates fit naturally as "
        f"inbound or outbound, return an empty array for that direction — do not force connections.\n\n"
        f'Respond with this JSON structure only:\n'
        f'{{"narrative": "...", "inbound": [{{"id": 42, "title": "..."}}], "outbound": [{{"id": 17, "title": "..."}}]}}'
    )

    print(f"[progression] job_id={job_id} title={job['title']!r} candidates={len(candidates)}", flush=True)

    # Step 5 — Call Sonnet
    try:
        resp = httpx.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      SONNET_MODEL,
                "max_tokens": 1000,
                "system":     PROGRESSION_SYSTEM_PROMPT,
                "messages":   [{"role": "user", "content": user_prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        result_text = resp.json()["content"][0]["text"].strip()
        # Strip markdown code fences if present
        if result_text.startswith("```"):
            result_text = result_text[result_text.find("\n")+1:]
            if result_text.endswith("```"):
                result_text = result_text[:-3].rstrip()
        result = json.loads(result_text)
    except Exception as e:
        print(f"[progression] Sonnet call failed ({e})", flush=True)
        jobs_conn.close()
        return jsonify({"has_progression": False})

    # Step 6 — Write to cache
    try:
        jobs_conn.execute(
            "INSERT OR REPLACE INTO job_progression_cache "
            "(job_id, narrative, inbound_json, outbound_json, prompt_version, created_at) "
            "VALUES (?, ?, ?, ?, 4, ?)",
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
        cid  = c["id"]
        meta = course_meta_by_id.get(cid)
        if not meta:
            continue
        db = course_row(cid)
        results.append(format_course(meta, db, course_score_by_id[cid]))

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
    """Create analytics.db and events table if they don't exist (first-run on fresh deploy)."""
    try:
        conn = sqlite3.connect(ANALYTICS_DB)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, ts TEXT NOT NULL, event TEXT NOT NULL, "
            "entity_type TEXT, entity_id INTEGER, entity_title TEXT, meta TEXT)"
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

_init_analytics_db()


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
