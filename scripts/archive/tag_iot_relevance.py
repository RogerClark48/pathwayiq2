"""
tag_iot_relevance.py — Bulk Haiku inference pass to classify job records
for relevance to Institutes of Technology.

Adds iot_relevant and iot_relevance_note columns to the jobs table in
job_roles_asset.db. Safe to re-run if interrupted — skips rows already tagged.

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/tag_iot_relevance.py
"""

import json
import re
import sqlite3
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "job_roles_asset.db"

load_dotenv(PROJECT_ROOT / ".env")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 100
DELAY = 0.5  # seconds between API calls

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are classifying job roles for relevance to Institutes of Technology (IoTs) — further education institutions focused on high-priority STEM and technical sectors. IoTs collectively cover the following areas:

- Engineering and advanced manufacturing: aerospace, automotive, high-value manufacturing
- Digital and information technology: software development, cyber security, cloud computing, AI, data analytics
- Construction and the built environment: civil engineering, modern methods of construction, sustainable building
- Energy and clean tech: renewable energy, nuclear power, electric vehicles
- Healthcare and life sciences: medical engineering, health technology, pharmaceutical sciences (technical roles, not direct patient care)
- Transport and logistics: maritime technology, aviation, logistics management
- Agri-tech: precision agriculture, sustainable farming technology
- Creative and media technologies: digital media, game design, technical creative roles
- Technical business skills: project management, FinTech, technical operations

The key distinction is technical depth. A medical device engineer is IoT territory; a general practice nurse is not. A FinTech developer is IoT territory; a general accountant is not. Each IoT adapts to its local economy, so scope is broad across the national network.

For each job, assess whether it represents a career outcome a student at such an institution might reasonably aspire to, or a role that a course at such an institution might lead toward.

Respond with JSON only. No preamble. No markdown. Format:
{"relevance": "<high|medium|low|no>", "note": "<one sentence reason, max 15 words>"}

Relevance levels:
- high: clearly within IoT territory — core technical, engineering, digital, or STEM role
- medium: adjacent or plausibly connected — when in doubt, prefer this over low
- low: only where the connection is a genuine stretch
- no: clearly outside IoT scope — e.g. general retail, hospitality, primary healthcare, pure arts. Use sparingly."""


def build_user_message(title: str, overview: str, entry_routes: str | None) -> str:
    msg = f"Job title: {title}\n\n{overview}"
    if entry_routes:
        msg += f"\n\nEntry routes: {entry_routes}"
    return msg


# ---------------------------------------------------------------------------
# JSON parsing — handles markdown fences if Haiku wraps its response
# ---------------------------------------------------------------------------
def parse_response(raw: str) -> tuple[str, str]:
    """
    Returns (relevance, note). On failure returns ('error', raw_response).
    """
    text = raw.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        relevance = str(data.get("relevance", "error")).lower()
        note = str(data.get("note", ""))
        if relevance not in ("high", "medium", "low", "no"):
            return "error", f"unexpected relevance value: {relevance}"
        return relevance, note
    except json.JSONDecodeError:
        return "error", raw[:200]  # store truncated raw response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    client = anthropic.Anthropic()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Add columns if they don't already exist
    for ddl in [
        "ALTER TABLE jobs ADD COLUMN iot_relevant TEXT",
        "ALTER TABLE jobs ADD COLUMN iot_relevance_note TEXT",
    ]:
        try:
            cur.execute(ddl)
            conn.commit()
            print(f"Column added: {ddl.split('ADD COLUMN')[1].strip().split()[0]}")
        except Exception:
            pass  # column already exists

    # Count total jobs for progress display
    cur.execute("SELECT COUNT(*) FROM jobs")
    total_jobs = cur.fetchone()[0]

    # Fetch untagged records that have content
    cur.execute("""
        SELECT id, title, overview, entry_routes
        FROM jobs
        WHERE iot_relevant IS NULL
          AND overview IS NOT NULL
        ORDER BY id
    """)
    rows = cur.fetchall()

    skipped = total_jobs - len(rows)
    print(f"\nTotal jobs: {total_jobs}")
    print(f"Already tagged or no content (skipping): {skipped}")
    print(f"To classify: {len(rows)}\n")

    counts = {"high": 0, "medium": 0, "low": 0, "no": 0, "error": 0}
    processed = 0

    for row in rows:
        job_id = row["id"]
        title = row["title"] or ""
        overview = row["overview"] or ""
        entry_routes = row["entry_routes"]  # may be None

        user_message = build_user_message(title, overview, entry_routes)

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
        except Exception as e:
            raw = f"API error: {e}"
            relevance, note = "error", raw[:200]
        else:
            relevance, note = parse_response(raw)

        cur.execute(
            "UPDATE jobs SET iot_relevant = ?, iot_relevance_note = ? WHERE id = ?",
            (relevance, note, job_id),
        )
        conn.commit()

        counts[relevance] = counts.get(relevance, 0) + 1
        processed += 1
        print(f"[{processed:4d}/{len(rows)}] {title[:55]:<55} → {relevance}")

        if processed < len(rows):
            time.sleep(DELAY)

    conn.close()

    # Count truly skipped (NULL overview)
    conn2 = sqlite3.connect(DB_PATH)
    cur2 = conn2.cursor()
    cur2.execute("SELECT COUNT(*) FROM jobs WHERE overview IS NULL")
    null_content = cur2.fetchone()[0]
    conn2.close()

    print(f"\nClassification complete.")
    print(f"  high:    {counts.get('high', 0)}")
    print(f"  medium:  {counts.get('medium', 0)}")
    print(f"  low:     {counts.get('low', 0)}")
    print(f"  no:      {counts.get('no', 0)}")
    print(f"  error:   {counts.get('error', 0)}")
    print(f"  skipped: {null_content}  (no content)")


if __name__ == "__main__":
    main()
