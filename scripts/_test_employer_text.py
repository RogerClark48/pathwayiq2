"""
_test_employer_text.py — 4-job test for employer display text generation.
Run specific job IDs for approval before the full 274 pass.
"""
import json
import re
import sqlite3

import anthropic
from dotenv import load_dotenv

load_dotenv()

LMI_DB  = r"C:\Dev\pathwayiq2\lmi.db"
JOBS_DB = r"C:\Dev\pathwayiq2\job_roles_asset.db"

lmi  = sqlite3.connect(LMI_DB)
jobs = sqlite3.connect(JOBS_DB)
lmi.row_factory  = sqlite3.Row
jobs.row_factory = sqlite3.Row
client = anthropic.Anthropic()

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
    "region", "area", "borough", "sector", "industries", "industry", "digital",
    "creative", "tech", "technology", "services", "professional", "business",
    "health", "care", "science", "advanced", "manufacturing", "construction",
    "built", "environment", "clean", "growth", "net", "zero", "innovation",
    "life", "cyber", "ai", "public", "private", "voluntary", "local", "national",
    "regional", "global", "large", "small", "major", "key", "main", "core",
    "central", "northern", "quarter", "square", "city", "centre", "hub", "park",
    "campus", "trust", "authority", "combined", "foundation", "university",
    "college", "social", "media", "production",
    "firms", "companies", "organisations", "employers", "providers",
    "contractors", "agencies", "institutes",
}


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


def allowed_names(mappings: list) -> set[str]:
    names: set[str] = set()
    for m in mappings:
        employers = json.loads(m["anchor_employers"]) if m["anchor_employers"] else []
        for e in employers:
            names.add(e.lower())
            # Split on spaces AND hyphens so "Mettler-Toledo Safeline" → "mettler", "toledo", "safeline"
            for word in re.split(r"[\s\-–—\.]+", e):
                names.add(word.lower().rstrip(".,;()"))
        for token in re.split(r"[\s,;()/\-–—]+", m["geography"] or ""):
            if token:
                names.add(token.lower().rstrip(".,;"))
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


def generate(job_id: int) -> tuple[str, str, int, list[str]]:
    j = jobs.execute(
        "SELECT id, title, overview, typical_duties FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    mappings = lmi.execute(
        """SELECT rc.is_primary, rc.confidence, c.cluster_name, c.description,
                  c.anchor_employers, c.geography, c.cross_cutting
           FROM role_clusters rc JOIN clusters c ON rc.cluster_id = c.cluster_id
           WHERE rc.job_id = ? ORDER BY rc.is_primary DESC""",
        (job_id,),
    ).fetchall()

    cluster_block = build_cluster_block(mappings)
    allowed = allowed_names(mappings)
    # Job title words are always allowed — they appear naturally in the output
    for word in re.split(r"\W+", j["title"] or ""):
        if word:
            allowed.add(word.lower())
    user_msg = f"Role: {j['title']}\nOverview: {(j['overview'] or '')[:300]}\n\n{cluster_block}"

    text = ""
    for attempt in range(2):
        suffix = "" if attempt == 0 else " Keep strictly to 90 words or fewer."
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0.3,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_msg + suffix}],
        )
        text = resp.content[0].text.strip()
        if len(text.split()) <= 140:
            break

    breaches = fence_check(text, allowed)
    return j["title"], text, len(text.split()), breaches


if __name__ == "__main__":
    test_jobs = [
        (775,  "single-cluster — adult nurse"),
        (651,  "multi-cluster — social media manager"),
        (1044, "cross-cutting — management consultant"),
        (922,  "review-flagged — electronics engineer"),
    ]

    for job_id, label in test_jobs:
        title, text, wc, breaches = generate(job_id)
        print(f"--- [{label}] --- {wc} words")
        print()
        print(text)
        print()
        if breaches:
            print(f"FENCE BREACH: {breaches}")
        else:
            print("Fence: clean")
        print()
        print("=" * 60)
        print()
