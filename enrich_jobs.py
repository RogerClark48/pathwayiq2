import sqlite3
import anthropic
import httpx
import os
import time
from dotenv import load_dotenv

load_dotenv()

DB_PATH = r"C:\Dev\pathwayiq\job_roles_asset.db"

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Cost tracking (claude-haiku-4-5 pricing)
# Input: $0.80 per million tokens
# Output: $8.00 per million tokens
total_input_tokens = 0
total_output_tokens = 0

def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * 0.80
    output_cost = (output_tokens / 1_000_000) * 8.00
    return input_cost + output_cost

DESCRIPTION_PROMPT = """You are helping build a careers advisory system. 
Based on the job profile page content below, write a standardised job description 
with exactly these sections:

OVERVIEW: 2-3 sentences summarising what this job involves and who it suits.

TYPICAL DUTIES: 4-6 bullet points of day-to-day tasks.

SKILLS REQUIRED: 4-6 bullet points of key skills and qualities needed.

ENTRY ROUTES: 2-3 sentences on typical qualifications and routes into the role.

SALARY RANGE: One line with typical salary range in the UK.

CAREER PROGRESSION: 2-3 sentences on where this role can lead.

Be factual and concise. Use plain English suitable for a careers advisor.
Do not invent information not present in the source content.

If the content does not contain recognisable job profile information, 
reply with exactly: INSUFFICIENT_CONTENT

Page content:
{content}"""

def fetch_page(url):
    """Fetch page with up to 3 retries."""
    for attempt in range(3):
        try:
            response = httpx.get(url, timeout=15, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; PathwayIQ/1.0)"})
            if response.status_code == 200 and len(response.text) > 2000:
                return response.text
            else:
                print(f"    Attempt {attempt+1}: status {response.status_code} or thin content ({len(response.text)} chars)")
        except Exception as e:
            print(f"    Attempt {attempt+1}: fetch error - {e}")
        
        if attempt < 2:
            print(f"    Retrying in 3 seconds...")
            time.sleep(3)
    
    return None

import re

def clean_html(html):
    """Strip scripts, styles and tags, leaving just text content."""
    # Remove script tags and contents
    html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL)
    # Remove style tags and contents
    html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL)
    # Remove all remaining HTML tags
    html = re.sub(r'<[^>]+>', ' ', html)
    # Collapse whitespace
    html = re.sub(r'\s+', ' ', html).strip()
    return html

def generate_description(page_content):
    """Send content to Claude and validate the response."""
    # Clean HTML before sending
    cleaned = clean_html(page_content)
    # Now take first 8000 chars of actual text content
    trimmed = cleaned[:8000]
    
    try:
        message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=600,
    messages=[{
        "role": "user",
        "content": DESCRIPTION_PROMPT.format(content=trimmed)
    }])
        global total_input_tokens, total_output_tokens
        total_input_tokens += message.usage.input_tokens
        total_output_tokens += message.usage.output_tokens
        
        result = message.content[0].text.strip()
        
        if "INSUFFICIENT_CONTENT" in result:
            return None, "insufficient_content"
        
        required_sections = ["OVERVIEW", "TYPICAL DUTIES", "SKILLS REQUIRED", "ENTRY ROUTES"]
        missing = [s for s in required_sections if s not in result.upper()]
        if missing:
            return None, f"missing_sections:{','.join(missing)}"
        
        return result, "ok"
        
    except Exception as e:
        return None, f"api_error:{e}"

def enrich_jobs(limit=10, source_filter=None):
    conn = sqlite3.connect(DB_PATH)
    
    # Add columns if they don't exist
    for column in ["enriched_description TEXT", "enrichment_status TEXT"]:
        try:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column}")
            conn.commit()
        except:
            pass
    
    # Build query
    query = """
        SELECT id, title, url, source 
        FROM jobs 
        WHERE is_active = 1
        AND url IS NOT NULL
        AND (enriched_description IS NULL OR enriched_description = '')
        AND (enrichment_status IS NULL OR enrichment_status = '')
    """
    if source_filter:
        query += f" AND source = '{source_filter}'"
    query += f" LIMIT {limit}"
    
    rows = conn.execute(query).fetchall()
    print(f"Found {len(rows)} jobs to enrich\n")
    
    success = 0
    failed = 0
    insufficient = 0
    
    for i, (job_id, title, url, source) in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] {title} ({source})")
        
        # Fetch page with retries
        page_content = fetch_page(url)
        
        if not page_content:
            print(f"    FAILED - could not fetch page after 3 attempts")
            conn.execute(
                "UPDATE jobs SET enrichment_status = 'fetch_failed' WHERE id = ?",
                (job_id,)
            )
            conn.commit()
            failed += 1
            time.sleep(1)
            continue
        
        # Generate description
        print(f"    Generating description...")
        description, status = generate_description(page_content)
        
        if status == "ok":
            conn.execute(
                """UPDATE jobs SET enriched_description = ?, 
                   enrichment_status = 'ok' WHERE id = ?""",
                (description, job_id)
            )
            print(f"    OK")
            success += 1
        elif status == "insufficient_content":
            print(f"    ALERT - Claude reported insufficient content")
            conn.execute(
                "UPDATE jobs SET enrichment_status = 'insufficient_content' WHERE id = ?",
                (job_id,)
            )
            insufficient += 1
        else:
            print(f"    ALERT - {status}")
            conn.execute(
                f"UPDATE jobs SET enrichment_status = ? WHERE id = ?",
                (status, job_id)
            )
            failed += 1
        
        conn.commit()
        time.sleep(1)
    total_cost = calculate_cost(total_input_tokens, total_output_tokens)
    print(f"\n  Input tokens:  {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.4f} (£{total_cost * 0.79:.4f})")
    conn.close()

        
    print(f"\n{'='*40}")
    print(f"Complete.")
    print(f"  Success:             {success}")
    print(f"  Insufficient content:{insufficient}")
    print(f"  Failed:              {failed}")
    print(f"\nTo review problem records:")
    print(f"  UPDATE peek.py to query WHERE enrichment_status != 'ok'")

enrich_jobs(limit=1252)


# ─── Stage: level tagging ─────────────────────────────────────────────────────

LEVEL_SYSTEM_PROMPT = """You are a UK careers adviser assigning RQF (Regulated Qualifications Framework) levels to job roles.

RQF level mapping:
- Level 1: Entry level — "no qualifications required", "no experience necessary", "will train", "no formal requirements"
- Level 2: GCSE / Intermediate — "GCSEs", "level 2", "some qualifications", "secondary school"
- Level 3: A Level / T Level / Advanced — "A levels", "T Level", "level 3", "advanced apprenticeship", "BTEC"
- Level 4: HNC / Higher — "HNC", "level 4", "higher apprenticeship", "higher national"
- Level 5: HND / Foundation Degree — "HND", "foundation degree", "level 5"
- Level 6: Bachelor's degree — "degree", "bachelor's", "graduate", "level 6", "undergraduate"
- Level 7: Master's / Postgraduate / Chartered — "master's", "postgraduate", "level 7", "chartered", "PhD", "doctorate"

Judgement rules:
- Where a range is stated ("a degree or relevant experience"), take the qualification route — assign the level of the qualification mentioned.
- Where multiple entry levels are mentioned, take the MINIMUM entry level — the lowest qualification that opens the door.
- entry_routes is the primary source. Use qualifications_summary to confirm or refine. Use progression as supporting context only, not as the basis for assignment.
- If the text is genuinely ambiguous or too thin to assign confidently, assign the closest reasonable level rather than returning nothing.

Respond with a single integer (1–7) only. No explanation, no text, just the number."""

# Level-tagging token tracking (separate from enrichment pass)
level_input_tokens = 0
level_output_tokens = 0


def _assign_level_haiku(entry_routes, qualifications_summary, progression):
    """Call Claude Haiku to assign RQF level 1–7. Returns integer or None on error."""
    global level_input_tokens, level_output_tokens

    parts = []
    if entry_routes:
        parts.append(f"Entry routes: {entry_routes}")
    if qualifications_summary:
        parts.append(f"Qualifications summary: {qualifications_summary}")
    if progression:
        parts.append(f"Progression: {progression}")

    user_content = "\n\n".join(parts)

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=LEVEL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}]
        )
        level_input_tokens += message.usage.input_tokens
        level_output_tokens += message.usage.output_tokens

        raw = message.content[0].text.strip()
        match = re.search(r'[1-7]', raw)
        if match:
            return int(match.group())
        print(f"    Unexpected response from Haiku: {raw!r}")
        return None
    except Exception as e:
        print(f"    API error: {e}")
        return None


def tag_levels():
    global level_input_tokens, level_output_tokens
    level_input_tokens = 0
    level_output_tokens = 0

    conn = sqlite3.connect(DB_PATH)

    # Add column if absent
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN level INTEGER")
        conn.commit()
        print("Added 'level' column.")
    except Exception:
        pass  # already exists

    # Keep the first 600 assignments (by id order); clear everything else
    conn.execute("""
        UPDATE jobs SET level = NULL
        WHERE id NOT IN (SELECT id FROM jobs ORDER BY id LIMIT 600)
    """)
    conn.commit()
    print("Reset level to NULL for records outside first 600.")

    rows = conn.execute(
        """SELECT id, entry_routes, qualifications_summary, progression
           FROM jobs
           WHERE level IS NULL"""
    ).fetchall()
    total = len(rows)
    print(f"Assigning levels for {total} records via Claude Haiku...\n")

    BATCH_SIZE = 100
    null_count = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start:batch_start + BATCH_SIZE]
        updates = []

        for job_id, entry_routes, qual_summary, progression in batch:
            if not any([entry_routes, qual_summary, progression]):
                updates.append((None, job_id))
                null_count += 1
                continue

            level = _assign_level_haiku(entry_routes, qual_summary, progression)
            updates.append((level, job_id))
            time.sleep(0.05)  # gentle pacing; increase if rate-limited

        conn.executemany("UPDATE jobs SET level = ? WHERE id = ?", updates)
        conn.commit()

        done = min(batch_start + BATCH_SIZE, total)
        cost_so_far = calculate_cost(level_input_tokens, level_output_tokens)
        print(f"  Batch {batch_start // BATCH_SIZE + 1}: "
              f"records {batch_start + 1}–{done} committed  "
              f"(cost so far: ${cost_so_far:.4f})")

    # Distribution report
    print("\nLevel distribution:")
    print(f"  {'Level':<8} {'Count':>6}")
    print("  " + "-" * 16)
    dist = conn.execute(
        "SELECT level, COUNT(*) FROM jobs GROUP BY level ORDER BY level"
    ).fetchall()
    total_all = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    for lvl, count in dist:
        label = str(lvl) if lvl is not None else "NULL"
        print(f"  {label:<8} {count:>6}  ({count / total_all * 100:.1f}%)")

    total_cost = calculate_cost(level_input_tokens, level_output_tokens)
    print(f"\nToken usage (level tagging):")
    print(f"  Input tokens:  {level_input_tokens:,}")
    print(f"  Output tokens: {level_output_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.4f} (£{total_cost * 0.79:.4f})")
    print(f"  NULL (no content): {null_count}")

    conn.close()
    print("\nDone.")


tag_levels()