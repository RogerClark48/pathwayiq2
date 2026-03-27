"""
embed_gmiot.py
Composes text chunks from named fields, embeds with Voyage AI voyage-3.5,
and loads into two new Chroma collections:
  - gmiot_courses  (83 courses × 2 chunks = 166)
  - gmiot_jobs     (1,216 jobs × 2 chunks = 2,432)

Existing EMIOT collections ('courses', 'jobs') are left untouched.
Stage 6 will clean them up once the app swap is confirmed.

Usage:
    python embed_gmiot.py            # full run
    python embed_gmiot.py --courses  # courses phase only
    python embed_gmiot.py --jobs     # jobs phase only
"""

import argparse
import os
import sqlite3
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = Path(__file__).parent
GMIOT_DB    = BASE_DIR / "gmiot.sqlite"
JOBS_DB     = BASE_DIR / "emiot_jobs_asset.db"
CHROMA_PATH = str(BASE_DIR / "chroma_store")

VOYAGE_MODEL     = "voyage-3.5"
VOYAGE_DIMS      = 1024
BATCH_SIZE       = 64

vo = voyageai.Client()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str], input_type: str = "document") -> list[list[float]]:
    result = vo.embed(
        texts,
        model=VOYAGE_MODEL,
        input_type=input_type,
        output_dimension=VOYAGE_DIMS,
    )
    return result.embeddings


def embed_in_batches(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        all_embeddings.extend(embed_batch(batch))
    return all_embeddings


# ---------------------------------------------------------------------------
# NULL sentinel helpers
# ---------------------------------------------------------------------------

def int_or_zero(val) -> int:
    """Chroma metadata does not accept None — use 0 as null sentinel for integers."""
    return int(val) if val is not None else 0


def str_or_empty(val) -> str:
    return str(val) if val is not None else ""


# ---------------------------------------------------------------------------
# Phase 1 — GMIoT courses
# ---------------------------------------------------------------------------

def embed_courses(chroma: chromadb.PersistentClient) -> None:
    print("\nPhase 1: Embedding GMIoT courses...")

    col = chroma.get_or_create_collection(
        name="gmiot_courses",
        metadata={"hnsw:space": "cosine"},
    )

    if col.count() > 0:
        print(f"  gmiot_courses already has {col.count()} chunks — skipping phase 1")
        print(f"  (delete the collection and re-run to rebuild)")
        return

    conn = sqlite3.connect(GMIOT_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM gmiot_courses ORDER BY course_id").fetchall()
    conn.close()

    ids, texts, metadatas = [], [], []
    skipped = []

    for row in rows:
        cid = str(row["course_id"])

        # Validate required fields
        if not row["overview"] or not row["what_you_will_learn"]:
            print(f"  WARNING: course_id={cid} has NULL content fields — skipped")
            skipped.append(cid)
            continue

        meta_base = {
            "course_id":         cid,
            "course_name":       str_or_empty(row["course_title"]),
            "provider":          str_or_empty(row["provider"]),
            "subject_area":      str_or_empty(row["subject_area"]),
            "ssa_category":      str_or_empty(row["ssa_label"]),
            "level":             int_or_zero(row["level"]),
            "qualification_type": str_or_empty(row["qual_type"]),
            "mode":              str_or_empty(row["mode"]),
            "url":               str_or_empty(row["course_url"]),
        }

        # _overview chunk
        ids.append(f"{cid}_overview")
        texts.append(
            f"{row['course_title']}\n"
            f"{row['overview']}\n"
            f"{row['what_you_will_learn']}"
        )
        metadatas.append({**meta_base, "chunk": "overview"})

        # _skills chunk
        ids.append(f"{cid}_skills")
        texts.append(
            f"{row['course_title']}\n"
            f"{row['entry_requirements']}\n"
            f"{row['progression']}"
        )
        metadatas.append({**meta_base, "chunk": "skills"})

    print(f"  Composing {len(texts)} chunks from {len(rows)} records"
          + (f" ({len(skipped)} skipped)" if skipped else "") + "...")

    embeddings = embed_in_batches(texts)

    print(f"  Adding to Chroma collection gmiot_courses...")
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"  Done. {len(ids)} chunks added.")


# ---------------------------------------------------------------------------
# Phase 2 — Jobs
# ---------------------------------------------------------------------------

def embed_jobs(chroma: chromadb.PersistentClient) -> None:
    print("\nPhase 2: Embedding jobs...")

    col = chroma.get_or_create_collection(
        name="gmiot_jobs",
        metadata={"hnsw:space": "cosine"},
    )

    if col.count() > 0:
        print(f"  gmiot_jobs already has {col.count()} chunks — skipping phase 2")
        print(f"  (delete the collection and re-run to rebuild)")
        return

    conn = sqlite3.connect(JOBS_DB)
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute(
        "SELECT id, title, source, url, salary_min, salary_max, salary_currency, "
        "overview, typical_duties, skills_required, entry_routes, progression "
        "FROM jobs ORDER BY id"
    ).fetchall()
    conn.close()

    ids, texts, metadatas = [], [], []
    skipped = 0

    for row in all_rows:
        if not row["overview"]:
            skipped += 1
            continue

        jid = str(row["id"])

        meta_base = {
            "job_id":          jid,
            "title":           str_or_empty(row["title"]),
            "chunk":           "",          # set per chunk below
            "source":          str_or_empty(row["source"]),
            "salary_min":      int_or_zero(row["salary_min"]),
            "salary_max":      int_or_zero(row["salary_max"]),
            "salary_currency": str_or_empty(row["salary_currency"]),
            "url":             str_or_empty(row["url"]),
        }

        # _overview chunk
        ids.append(f"{jid}_overview")
        texts.append(
            f"{row['title']}\n"
            f"{row['overview']}\n"
            f"{row['typical_duties']}"
        )
        metadatas.append({**meta_base, "chunk": "overview"})

        # _skills chunk
        ids.append(f"{jid}_skills")
        texts.append(
            f"{row['title']}\n"
            f"{row['skills_required']}\n"
            f"{row['entry_routes']}\n"
            f"{row['progression']}"
        )
        metadatas.append({**meta_base, "chunk": "skills"})

    processed = len(all_rows) - skipped
    print(f"  Composing {len(texts)} chunks from {processed} records"
          + (f" ({skipped} skipped — no overview)" if skipped else "") + "...")

    embeddings = embed_in_batches(texts)

    print(f"  Adding to Chroma collection gmiot_jobs...")
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"  Done. {len(ids)} chunks added.")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(chroma: chromadb.PersistentClient) -> None:
    print("\n--- Verification ---")
    for name in ("gmiot_courses", "gmiot_jobs"):
        try:
            col = chroma.get_collection(name)
            print(f"  {name}: {col.count()} chunks")

            sample = col.get(limit=1)
            if sample["ids"]:
                print(f"    sample id       : {sample['ids'][0]}")
                print(f"    sample doc[:120]: {sample['documents'][0][:120]!r}")
                print(f"    sample metadata : {sample['metadatas'][0]}")
        except Exception as e:
            print(f"  {name}: ERROR — {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed GMIoT courses and jobs into Chroma")
    parser.add_argument("--courses", action="store_true", help="Run courses phase only")
    parser.add_argument("--jobs",    action="store_true", help="Run jobs phase only")
    args = parser.parse_args()

    run_courses = args.courses or not (args.courses or args.jobs)
    run_jobs    = args.jobs    or not (args.courses or args.jobs)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if run_courses:
        embed_courses(client)
    if run_jobs:
        embed_jobs(client)

    verify(client)
    print("\nComplete.")


if __name__ == "__main__":
    main()
