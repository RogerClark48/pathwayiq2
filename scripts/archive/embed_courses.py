import re
import sqlite3
import chromadb
import ollama

DB_PATH = r"C:\Dev\pathwayiq\emiot.sqlite"
CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"

TEST_MODE = False  # Set True to limit to 10 records and print parse debug for the first one

SECTION_NAMES = [
    "OVERVIEW",
    "WHAT YOU WILL LEARN",
    "ENTRY REQUIREMENTS",
    "QUALIFICATION AWARDED",
    "QUALIFICATION TYPE",
    "PROGRESSION",
    "SSA CATEGORY",
]

GROUP_1 = ["OVERVIEW", "WHAT YOU WILL LEARN"]
GROUP_2 = ["ENTRY REQUIREMENTS", "QUALIFICATION AWARDED", "PROGRESSION"]


def normalize_text(text):
    """Strip markdown heading markers so all headers are plain text."""
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text


def parse_sections(text):
    """Return dict of SECTION_NAME -> body text."""
    text = normalize_text(text)

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
        SELECT courseId, courseName, provider, level, subjectArea,
               courseUrl, description, qualificationType, ssa_category
        FROM Course
        WHERE enrichment_status = 'ok'
        AND description IS NOT NULL
        AND description != ''
    """).fetchall()
    conn.close()

    if TEST_MODE:
        rows = rows[:10]
        print(f"TEST MODE: capped at {len(rows)} records")
    else:
        print(f"Loaded {len(rows)} enriched courses")

    # --- parse debug for first record ---
    if TEST_MODE and rows:
        first = rows[0]
        print(f"\n=== PARSE DEBUG: {first[1]} ===")
        print("--- raw description ---")
        print(first[6])
        print("\n--- parsed sections ---")
        sections = parse_sections(first[6])
        for name, body in sections.items():
            print(f"\n[{name}]\n{body}")
        print("\n--- chunk 1 (OVERVIEW + WHAT YOU WILL LEARN) ---")
        print(build_chunk(sections, GROUP_1))
        print("\n--- chunk 2 (ENTRY REQUIREMENTS + QUALIFICATION AWARDED + PROGRESSION) ---")
        print(build_chunk(sections, GROUP_2))
        print("=" * 60 + "\n")

    print("Setting up Chroma collection...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("courses")
        print("Deleted existing courses collection")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="courses",
        metadata={"hnsw:space": "cosine"},
    )

    skipped = 0
    stored = 0

    print("Embedding courses...")
    for i, (course_id, name, provider, level, subject_area,
            url, description, qualification_type, ssa_category) in enumerate(rows):

        sections = parse_sections(description)

        chunk1 = build_chunk(sections, GROUP_1)
        chunk2 = build_chunk(sections, GROUP_2)

        if not chunk1 and not chunk2:
            print(f"  [{i+1}/{len(rows)}] SKIP (no parseable sections): {name}")
            skipped += 1
            continue

        meta = {
            "course_id": str(course_id),
            "course_name": name or "",
            "provider": provider or "",
            "level": int(level) if level is not None else 0,
            "subject_area": subject_area or "",
            "qualification_type": qualification_type or "",
            "ssa_category": ssa_category or "",
            "url": url or "",
        }

        ids, embeddings, documents, metadatas = [], [], [], []

        if chunk1:
            ids.append(f"{course_id}_overview")
            embeddings.append(embed(chunk1))
            documents.append(chunk1)
            metadatas.append({**meta, "chunk": "overview"})

        if chunk2:
            ids.append(f"{course_id}_skills")
            embeddings.append(embed(chunk2))
            documents.append(chunk2)
            metadatas.append({**meta, "chunk": "skills"})

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        stored += len(ids)
        print(f"  [{i+1}/{len(rows)}] {name} ({len(ids)} chunks)")

    print(f"\nDone.")
    print(f"  Courses processed: {len(rows) - skipped}")
    print(f"  Courses skipped:   {skipped}")
    print(f"  Chunks stored:     {stored}")
    print(f"  Chroma path:       {CHROMA_PATH}")


if __name__ == "__main__":
    main()
