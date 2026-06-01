"""
build_employer_text.py — Generate employer display text for all carded jobs.

Reads cluster assignments from lmi.db, generates a 'where could this lead
in Greater Manchester?' paragraph per job via Haiku, stores results in
lmi.db.job_employer_text. Runs a mechanical fence check on every output.

Restartable: skips jobs already in job_employer_text unless --force is passed.

Usage:
    python scripts/build_employer_text.py
    python scripts/build_employer_text.py --force
"""

import json
import re
import sqlite3
import sys
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

LMI_DB  = r"C:\Dev\pathwayiq2\lmi.db"
JOBS_DB = r"C:\Dev\pathwayiq2\job_roles_asset.db"
FF_DB   = r"C:\Dev\pathwayiq2\futurefinder.sqlite"

HAIKU_MODEL    = "claude-haiku-4-5-20251001"
MAX_WORDS      = 140
PROMPT_VERSION = "1.0"  # bump when SYSTEM prompt changes to mark rows for regeneration

SYSTEM = (
    "You are writing a short 'where could this lead in Greater Manchester?' note "
    "for a prospective student looking at a job role. Write up to two short paragraphs, "
    "140 words maximum, plain everyday language suitable for someone aged 16+, "
    "mobile-first. Second person where natural.\n\n"
    "You are given the role and the Greater Manchester employer cluster(s) it maps to, "
    "each with named employers and a geographic concentration. Tell the user, usefully "
    "and concretely: the kinds of organisation that employ this role in Greater Manchester, "
    "a few named examples, and roughly where in the region they concentrate.\n\n"
    "Where the role spans more than one cluster, you may use a second paragraph for the "
    "secondary sector(s). For a single-cluster role, a second paragraph is optional — "
    "only use it if there is genuinely more useful detail, never to pad to length.\n\n"
    "Rules:\n"
    "- Named employers and specific places must come only from the cluster data provided. "
    "You may add general sector context where you are confident, but never name an employer, "
    "site, or location not in the cluster data.\n"
    "- Name 2-4 employers at most. Prefer the most recognisable.\n"
    "- Lead with the primary cluster. If a secondary cluster is given, add a brief clause "
    "showing the role's reach, but do not give equal weight to every cluster.\n"
    "- For cross-cutting clusters, name only the types of organisation and any specific "
    "names present in the cluster data (e.g. GMCA). Never introduce company names, "
    "local authority names, or university names from your own knowledge — even plausible "
    "ones. If the cluster lists no specific employers, describe the kinds of organisation "
    "instead (e.g. 'local authorities', 'universities', 'professional services firms'). "
    "Do not fill the gap with names you know.\n"
    "- Do not name specific places not present verbatim in the geography field. If the "
    "cluster geography is broad (e.g. 'all ten boroughs', 'right across the region'), "
    "reflect that breadth in your own words rather than listing specific place names.\n"
    "- No salary figures, no vacancy counts, no claims about how many jobs exist.\n"
    "- End cleanly. No encouragement padding.\n"
    "- Plain text only. No markdown, no bold, no bullet points."
)

SKIP_WORDS = {
    "greater", "manchester", "gm", "the", "you", "this", "if", "in", "and", "or",
    "of", "a", "an", "it", "is", "are", "to", "for", "with", "from", "on", "at",
    "by", "as", "its", "uk", "nhs", "there", "here", "beyond", "because", "across",
    "where", "these", "those", "that", "they", "their", "which", "your", "all",
    "but", "so", "can", "could", "not", "no", "any", "some", "many", "most",
    "more", "than", "been", "have", "has", "had", "will", "would", "may", "be",
    "do", "does", "did", "was", "were", "into", "over", "out", "up", "about",
    "like", "also", "both", "very", "just", "too", "only", "even", "well", "then",
    "when", "what", "how", "who", "role", "roles", "work", "based", "around",
    "region", "area", "borough", "boroughs", "sector", "industries", "industry",
    "digital", "creative", "tech", "technology", "services", "professional",
    "business", "health", "care", "science", "advanced", "manufacturing",
    "construction", "built", "environment", "clean", "growth", "net", "zero",
    "innovation", "life", "cyber", "ai", "public", "private", "voluntary", "local",
    "national", "regional", "global", "large", "small", "major", "key", "main",
    "core", "central", "northern", "quarter", "square", "city", "centre", "hub",
    "park", "campus", "trust", "authority", "combined", "foundation", "university",
    "college", "social", "media", "production", "firms", "companies", "organisations",
    "employers", "providers", "contractors", "agencies", "institutes",
    # Abbreviations and acronyms that are not employer names
    "saas", "tv", "av", "vr", "iot", "vfx", "ndt", "ai", "ar", "ict",
    "gp", "pr", "hr", "it", "odps", "copd", "nhs",
    # Common words the regex catches as capitalised
    "adult", "anchor", "every", "alternatively", "medical", "nursing",
    "nutritionists", "physiotherapists", "welders", "schools", "broadcast",
    "while", "big", "larger", "smaller", "research", "design", "with",
    "life", "sciences", "old", "young", "whether",
    "places", "because", "beyond", "whereas", "although", "however", "therefore",
    "including", "especially", "particularly", "generally", "typically", "often",
    "usually", "always", "never", "right", "real", "genuine", "strong", "growing",
    "thriving", "booming", "leading", "major", "significant", "dedicated",
}


def get_carded_job_ids() -> list[int]:
    conn = sqlite3.connect(FF_DB)
    rows = conn.execute(
        "SELECT curated_jobs FROM course_career_pathways WHERE curated_jobs IS NOT NULL"
    ).fetchall()
    conn.close()
    ids: set[int] = set()
    for (cj,) in rows:
        for entry in json.loads(cj):
            ids.add(entry["job_id"])
    return sorted(ids)


def build_cluster_block(mappings: list) -> str:
    lines = []
    for m in mappings:
        label = "PRIMARY CLUSTER" if m["is_primary"] else "SECONDARY CLUSTER"
        employers = json.loads(m["anchor_employers"]) if m["anchor_employers"] else []
        emp_str = "\n".join(employers) if employers else "(none listed — describe types of organisation only)"
        lines.append(f"{label}: {m['cluster_name']} (cross_cutting={m['cross_cutting']})")
        lines.append(f"Description: {m['description']}")
        lines.append(f"Anchor employers:\n{emp_str}")
        lines.append(f"Geography: {m['geography']}")
        lines.append("")
    return "\n".join(lines).strip()


def allowed_names(mappings: list, job_title: str) -> set[str]:
    names: set[str] = set()
    for m in mappings:
        employers = json.loads(m["anchor_employers"]) if m["anchor_employers"] else []
        for e in employers:
            names.add(e.lower())
            for word in re.split(r"[\s\-–—\./'']+", e):
                names.add(word.lower().rstrip(".,;()'’"))
        for token in re.split(r"[\s,;()/\-–—]+", m["geography"] or ""):
            if token:
                names.add(token.lower().rstrip(".,;"))
    for word in re.split(r"\W+", job_title or ""):
        if word:
            names.add(word.lower())
    return names


def fence_check(text: str, allowed: set[str]) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-zA-Z&]+(?:\s+[A-Z][a-zA-Z&]+)*\b", text)
    breaches = []
    for c in candidates:
        words = [w.lower().rstrip(".,;") for w in c.split()]
        non_skip = [w for w in words if w not in SKIP_WORDS]
        if not non_skip:
            continue
        if not any(w in allowed for w in non_skip):
            breaches.append(c)
    return breaches


def generate(
    client: anthropic.Anthropic,
    job_id: int,
    job_title: str,
    job_overview: str,
    mappings: list,
) -> tuple[str, list[str]]:
    cluster_block = build_cluster_block(mappings)
    allowed = allowed_names(mappings, job_title)
    user_msg = f"Role: {job_title}\nOverview: {(job_overview or '')[:300]}\n\n{cluster_block}"

    text = ""
    for attempt in range(2):
        suffix = "" if attempt == 0 else " Keep strictly to 140 words or fewer."
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.3,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_msg + suffix}],
        )
        text = resp.content[0].text.strip()
        if len(text.split()) <= MAX_WORDS:
            break

    breaches = fence_check(text, allowed)
    return text, breaches


def main():
    force = "--force" in sys.argv

    lmi  = sqlite3.connect(LMI_DB)
    jobs = sqlite3.connect(JOBS_DB)
    lmi.row_factory  = sqlite3.Row
    jobs.row_factory = sqlite3.Row

    # Create table
    lmi.execute("""
        CREATE TABLE IF NOT EXISTS job_employer_text (
            job_id         INTEGER PRIMARY KEY,
            employer_text  TEXT NOT NULL,
            fence_breaches TEXT,
            prompt_version TEXT NOT NULL,
            generated_at   TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    lmi.commit()

    if force:
        lmi.execute("DELETE FROM job_employer_text")
        lmi.commit()
        print("--force: cleared existing rows.", flush=True)

    already_done = {r[0] for r in lmi.execute("SELECT job_id FROM job_employer_text").fetchall()}

    job_ids = get_carded_job_ids()
    to_run  = [jid for jid in job_ids if jid not in already_done]
    print(f"{len(job_ids)} carded jobs — {len(already_done)} already done — {len(to_run)} to generate.", flush=True)

    client     = anthropic.Anthropic()
    all_breaches: list[tuple[int, str, list[str]]] = []
    done = 0

    for i, job_id in enumerate(to_run, 1):
        j = jobs.execute(
            "SELECT id, title, overview FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if not j:
            print(f"  [{i:>3}/{len(to_run)}] job {job_id} — not in jobs DB, skipping", flush=True)
            continue

        mappings = lmi.execute(
            """SELECT rc.is_primary, rc.confidence, c.cluster_name, c.description,
                      c.anchor_employers, c.geography, c.cross_cutting
               FROM role_clusters rc JOIN clusters c ON rc.cluster_id = c.cluster_id
               WHERE rc.job_id = ? ORDER BY rc.is_primary DESC""",
            (job_id,),
        ).fetchall()

        if not mappings:
            print(f"  [{i:>3}/{len(to_run)}] {j['title'][:40]} — no cluster mapping, skipping", flush=True)
            continue

        text, breaches = generate(client, job_id, j["title"], j["overview"], mappings)
        wc = len(text.split())

        lmi.execute(
            """INSERT OR REPLACE INTO job_employer_text
               (job_id, employer_text, fence_breaches, prompt_version)
               VALUES (?,?,?,?)""",
            (job_id, text, json.dumps(breaches) if breaches else None, PROMPT_VERSION),
        )
        lmi.commit()
        done += 1

        breach_flag = f"  FENCE:{breaches}" if breaches else ""
        print(
            f"  [{i:>3}/{len(to_run)}] {j['title'][:42]:<42} {wc:>3}w{breach_flag}",
            flush=True,
        )

        if breaches:
            all_breaches.append((job_id, j["title"], breaches))

        if i % 10 == 0:
            time.sleep(0.5)

    print(f"\nDone. {done} generated.", flush=True)

    total = lmi.execute("SELECT COUNT(*) FROM job_employer_text").fetchone()[0]
    print(f"Total in job_employer_text: {total}", flush=True)

    print("\nGenerated by primary cluster:", flush=True)
    rows = lmi.execute("""
        SELECT c.cluster_name, COUNT(*) as cnt
        FROM job_employer_text jt
        JOIN role_clusters rc ON rc.job_id = jt.job_id AND rc.is_primary = 1
        JOIN clusters c ON c.cluster_id = rc.cluster_id
        GROUP BY rc.cluster_id ORDER BY cnt DESC
    """).fetchall()
    for name, cnt in rows:
        print(f"  {cnt:>4}  {name}", flush=True)

    flagged = lmi.execute(
        "SELECT COUNT(*) FROM job_employer_text WHERE fence_breaches IS NOT NULL"
    ).fetchone()[0]
    print(f"\nFence flags stored: {flagged}", flush=True)
    if all_breaches:
        print("Jobs to review:", flush=True)
        for job_id, title, breaches in all_breaches:
            print(f"  [{job_id}] {title}: {breaches}", flush=True)
    else:
        print("No fence breaches detected.", flush=True)

    lmi.close()
    jobs.close()


if __name__ == "__main__":
    main()
