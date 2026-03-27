import re
import sqlite3
import chromadb
import ollama

DB_PATH = r"C:\Dev\pathwayiq\emiot_jobs_asset.db"
CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"

TEST_MODE = False  # Set True to limit to 10 records and print parse debug for the first one

SECTION_NAMES = [
    "OVERVIEW",
    "TYPICAL DUTIES",
    "SKILLS REQUIRED",
    "ENTRY ROUTES",
    "SALARY RANGE",
    "CAREER PROGRESSION",
]

GROUP_1 = ["OVERVIEW", "TYPICAL DUTIES"]
GROUP_2 = ["SKILLS REQUIRED", "ENTRY ROUTES", "CAREER PROGRESSION"]


def normalize_text(text):
    """Strip markdown heading markers so all headers are plain text."""
    # ## SECTION NAME  ->  SECTION NAME
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
    # **SECTION NAME**  ->  SECTION NAME  (bold markers)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text


def parse_sections(text):
    """Return dict of SECTION_NAME -> body text."""
    text = normalize_text(text)

    # Matches any known section header line with optional trailing colon
    header_pattern = re.compile(
        r"^(" + "|".join(re.escape(s) for s in SECTION_NAMES) + r"):?\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    sections = {}
    current_name = None
    current_lines = []

    for line in text.splitlines():
        m = header_pattern.match(line.strip())
        if m:
            if current_name is not None:
                sections[current_name] = "\n".join(current_lines).strip()
            current_name = m.group(1).upper()
            current_lines = []
        else:
            if current_name is not None:
                current_lines.append(line)

    if current_name is not None:
        sections[current_name] = "\n".join(current_lines).strip()

    return sections


def build_chunk(sections, group):
    """Concatenate named sections into a single string for embedding."""
    parts = []
    for name in group:
        body = sections.get(name, "").strip()
        if body:
            parts.append(f"{name}\n{body}")
    return "\n\n".join(parts)


def embed(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def main():
    print(f"{'[TEST MODE] ' if TEST_MODE else ''}Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, title, url, source, enriched_description,
               salary_min, salary_max, salary_currency
        FROM jobs
        WHERE enrichment_status = 'ok'
        AND enriched_description IS NOT NULL
        AND enriched_description != ''
    """).fetchall()
    conn.close()

    if TEST_MODE:
        rows = rows[:10]
        print(f"TEST MODE: capped at {len(rows)} records")
    else:
        print(f"Loaded {len(rows)} enriched jobs")

    # --- parse debug for first record ---
    if TEST_MODE and rows:
        first = rows[0]
        print(f"\n=== PARSE DEBUG: {first[1]} ===")
        print("--- raw enriched_description ---")
        print(first[4])
        print("\n--- parsed sections ---")
        sections = parse_sections(first[4])
        for name, body in sections.items():
            print(f"\n[{name}]\n{body}")
        print("\n--- chunk 1 (OVERVIEW + TYPICAL DUTIES) ---")
        print(build_chunk(sections, GROUP_1))
        print("\n--- chunk 2 (SKILLS REQUIRED + ENTRY ROUTES + CAREER PROGRESSION) ---")
        print(build_chunk(sections, GROUP_2))
        print("=" * 60 + "\n")

    print("Setting up Chroma collection...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("jobs")
        print("Deleted existing jobs collection")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="jobs",
        metadata={"hnsw:space": "cosine"},
    )

    skipped = 0
    stored = 0

    print("Embedding jobs...")
    for i, (job_id, title, url, source, raw_desc,
            salary_min, salary_max, salary_currency) in enumerate(rows):
        sections = parse_sections(raw_desc)

        chunk1 = build_chunk(sections, GROUP_1)
        chunk2 = build_chunk(sections, GROUP_2)

        if not chunk1 and not chunk2:
            print(f"  [{i+1}/{len(rows)}] SKIP (no parseable sections): {title}")
            skipped += 1
            continue

        meta = {
            "job_id": str(job_id),
            "title": title or "",
            "url": url or "",
            "source": source or "",
            "salary_min": float(salary_min) if salary_min is not None else 0.0,
            "salary_max": float(salary_max) if salary_max is not None else 0.0,
            "salary_currency": salary_currency or "",
        }

        ids, embeddings, documents, metadatas = [], [], [], []

        if chunk1:
            ids.append(f"{job_id}_overview")
            embeddings.append(embed(chunk1))
            documents.append(chunk1)
            metadatas.append({**meta, "chunk": "overview"})

        if chunk2:
            ids.append(f"{job_id}_skills")
            embeddings.append(embed(chunk2))
            documents.append(chunk2)
            metadatas.append({**meta, "chunk": "skills"})

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        stored += len(ids)
        print(f"  [{i+1}/{len(rows)}] {title} ({len(ids)} chunks)")

    print(f"\nDone.")
    print(f"  Jobs processed: {len(rows) - skipped}")
    print(f"  Jobs skipped:   {skipped}")
    print(f"  Chunks stored:  {stored}")
    print(f"  Chroma path:    {CHROMA_PATH}")


if __name__ == "__main__":
    main()
