import re
import sqlite3
import anthropic
import httpx
import os
import time
from dotenv import load_dotenv

load_dotenv()

DB_PATH = r"C:\Dev\pathwayiq\emiot.sqlite"

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Cost tracking (claude-haiku-4-5 pricing)
# Input: $0.80 per million tokens
# Output: $8.00 per million tokens
total_input_tokens = 0
total_output_tokens = 0


def calculate_cost(input_tokens, output_tokens):
    return (input_tokens / 1_000_000) * 0.80 + (output_tokens / 1_000_000) * 8.00


SSA_CATEGORIES = [
    "1 - Health, Public Services and Care",
    "2 - Science and Mathematics",
    "3 - Agriculture, Horticulture and Animal Care",
    "4 - Engineering and Manufacturing Technologies",
    "5 - Construction, Planning and the Built Environment",
    "6 - Digital Technology",
    "7 - Retail and Commercial Enterprise",
    "8 - Leisure, Travel and Tourism",
    "9 - Arts, Media and Publishing",
    "10 - History, Philosophy and Theology",
    "11 - Social Sciences",
    "12 - Languages, Literature and Culture",
    "13 - Education and Training",
    "14 - Preparation for Life and Work",
    "15 - Business, Administration and Law",
]

SSA_LIST = "\n".join(SSA_CATEGORIES)

DESCRIPTION_PROMPT = f"""You are helping build a course discovery system for learners and advisers.
Based on the course page content below, write a standardised course description
with exactly these sections:

OVERVIEW: 2-3 sentences summarising what this course covers and who it suits.

WHAT YOU WILL LEARN: 4-6 bullet points of the main topics or skills covered.

ENTRY REQUIREMENTS: 2-3 sentences on typical prerequisites, prior qualifications, or experience needed.

QUALIFICATION AWARDED: One line with the full qualification title gained on completion (e.g. BTEC Level 3 Extended Diploma in Engineering, T Level in Digital Production Design and Development).

QUALIFICATION TYPE: One word or short phrase giving just the qualification category (e.g. MSc, BA Hons, HNC, HND, T Level, BTEC, A Level, Access, Apprenticeship, Certificate, Diploma).

PROGRESSION: 2-3 sentences on where this course can lead (further study or employment).

SSA CATEGORY: Return ONLY one category exactly as written from this list:
{SSA_LIST}

Format each section with the heading on its own line, followed by a blank line, then the content below it.
Be factual and concise. Use plain English suitable for a learner or careers adviser.
Do not invent information not present in the source content.

If the content does not contain recognisable course information,
reply with exactly: INSUFFICIENT_CONTENT

Page content:
{{content}}"""


def fetch_page(url):
    for attempt in range(3):
        try:
            response = httpx.get(url, timeout=15, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; PathwayIQ/1.0)"})
            if response.status_code == 200 and len(response.text) > 500:
                return response.text
            else:
                print(f"    Attempt {attempt+1}: status {response.status_code} or thin content ({len(response.text)} chars)")
        except Exception as e:
            print(f"    Attempt {attempt+1}: fetch error - {e}")

        if attempt < 2:
            print(f"    Retrying in 3 seconds...")
            time.sleep(3)

    return None


def clean_html(html):
    html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL)
    html = re.sub(r'<[^>]+>', ' ', html)
    html = re.sub(r'\s+', ' ', html).strip()
    return html


def extract_ssa_category(text):
    """Pull the SSA CATEGORY value out of the generated description."""
    m = re.search(r'SSA CATEGORY:?\s*(.+)', text, flags=re.IGNORECASE)
    if not m:
        return None
    candidate = m.group(1).strip().strip('*').strip().rstrip('.')
    # Accept if it matches one of our known categories (case-insensitive)
    for cat in SSA_CATEGORIES:
        if cat.lower() == candidate.lower():
            return cat
    # Fallback: return whatever Claude gave us (trimmed)
    return candidate or None


def extract_qualification(text):
    """Pull the QUALIFICATION TYPE (short category) out of the generated description."""
    m = re.search(r'QUALIFICATION TYPE:?\s*(.+)', text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().strip('*').strip().rstrip('.')


def generate_description(page_content):
    global total_input_tokens, total_output_tokens

    cleaned = clean_html(page_content)
    trimmed = cleaned[:8000]

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": DESCRIPTION_PROMPT.format(content=trimmed)
            }]
        )
        total_input_tokens += message.usage.input_tokens
        total_output_tokens += message.usage.output_tokens

        result = message.content[0].text.strip()

        if TEST_MODE:
            print("\n--- RAW CLAUDE RESPONSE ---")
            print(result)
            print("--- END ---\n")

        if "INSUFFICIENT_CONTENT" in result:
            return None, None, None, "insufficient_content"

        required_sections = ["OVERVIEW", "WHAT YOU WILL LEARN", "ENTRY REQUIREMENTS",
                             "QUALIFICATION AWARDED", "QUALIFICATION TYPE", "PROGRESSION", "SSA CATEGORY"]
        missing = [s for s in required_sections if s not in result.upper()]
        if missing:
            return None, None, None, f"missing_sections:{','.join(missing)}"

        ssa = extract_ssa_category(result)
        qualification = extract_qualification(result)
        return result, ssa, qualification, "ok"

    except Exception as e:
        return None, None, None, f"api_error:{e}"


def enrich_courses(limit=None, source_filter=None):
    conn = sqlite3.connect(DB_PATH)

    # Add columns if they don't exist
    for col in ["enrichment_status TEXT", "ssa_category TEXT"]:
        try:
            conn.execute(f"ALTER TABLE Course ADD COLUMN {col}")
            conn.commit()
        except Exception:
            pass  # column already exists

    query = """
        SELECT courseId, courseName, provider, courseUrl
        FROM Course
        WHERE courseUrl IS NOT NULL
        AND (enrichment_status IS NULL OR enrichment_status = '')
    """
    if source_filter:
        query += f" AND provider = '{source_filter}'"
    if limit:
        query += f" LIMIT {limit}"

    rows = conn.execute(query).fetchall()
    print(f"Found {len(rows)} courses to enrich\n")

    success = 0
    failed = 0
    insufficient = 0

    for i, (course_id, name, provider, url) in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] {name} ({provider})")

        page_content = fetch_page(url)

        if not page_content:
            print(f"    FAILED - could not fetch page after 3 attempts")
            conn.execute(
                "UPDATE Course SET enrichment_status = 'fetch_failed' WHERE courseId = ?",
                (course_id,)
            )
            conn.commit()
            failed += 1
            time.sleep(1)
            continue

        print(f"    Generating description...")
        description, ssa, qualification, status = generate_description(page_content)

        if status == "ok":
            conn.execute(
                """UPDATE Course
                   SET description = ?,
                       ssa_category = ?,
                       qualificationType = COALESCE(NULLIF(?, ''), qualificationType),
                       enrichment_status = 'ok'
                   WHERE courseId = ?""",
                (description, ssa, qualification or '', course_id)
            )
            print(f"    OK  |  SSA: {ssa}  |  Qual: {qualification}")
            success += 1
        elif status == "insufficient_content":
            print(f"    ALERT - insufficient content")
            conn.execute(
                "UPDATE Course SET enrichment_status = 'insufficient_content' WHERE courseId = ?",
                (course_id,)
            )
            insufficient += 1
        else:
            print(f"    ALERT - {status}")
            conn.execute(
                "UPDATE Course SET enrichment_status = ? WHERE courseId = ?",
                (status, course_id)
            )
            failed += 1

        conn.commit()
        time.sleep(1)

    total_cost = calculate_cost(total_input_tokens, total_output_tokens)
    print(f"\n{'='*40}")
    print(f"Complete.")
    print(f"  Success:              {success}")
    print(f"  Insufficient content: {insufficient}")
    print(f"  Failed:               {failed}")
    print(f"\n  Input tokens:  {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.4f} (£{total_cost * 0.79:.4f})")
    conn.close()


TEST_MODE = False  # Set False to process all records

enrich_courses(limit=5 if TEST_MODE else None)
