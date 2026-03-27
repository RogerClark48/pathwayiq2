"""
embed_field_specific.py
-----------------------
Creates five new Chroma collections with single-field embeddings for
the match profile approach.

New collections (existing collections untouched):
  gmiot_courses_learning    -- course what_you_will_learn
  gmiot_courses_progression -- course progression
  gmiot_jobs_skills         -- job skills_required
  gmiot_jobs_duties         -- job typical_duties
  gmiot_jobs_entry          -- job entry_routes

Run:
  python embed_field_specific.py            # full run
  python embed_field_specific.py --courses  # course phases only
  python embed_field_specific.py --jobs     # job phases only
"""

import sys
import sqlite3
import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"
GMIOT_DB    = r"C:\Dev\pathwayiq\gmiot.sqlite"
JOBS_DB     = r"C:\Dev\pathwayiq\emiot_jobs_asset.db"
VOYAGE_MODEL = "voyage-3.5"
VOYAGE_DIMS  = 1024
BATCH_SIZE   = 64


# ---------------------------------------------------------------------------
# Voyage AI
# ---------------------------------------------------------------------------
vo = voyageai.Client()


def embed_batch(texts: list[str]) -> list[list[float]]:
    result = vo.embed(
        texts,
        model=VOYAGE_MODEL,
        input_type="document",
        output_dimension=VOYAGE_DIMS,
    )
    return result.embeddings


def embed_in_batches(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch     = texts[i : i + BATCH_SIZE]
        print(f"    Batch {batch_num}/{total_batches}...", end=" ", flush=True)
        all_embeddings.extend(embed_batch(batch))
    print()
    return all_embeddings


# ---------------------------------------------------------------------------
# Phase 1 — Course fields
# ---------------------------------------------------------------------------
def embed_course_fields(client: chromadb.PersistentClient) -> dict:
    print("\nPhase 1: Course field embeddings")

    conn = sqlite3.connect(GMIOT_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT course_id, course_title, what_you_will_learn, progression, "
        "ssa_label, provider, level, qual_type "
        "FROM gmiot_courses ORDER BY course_id"
    ).fetchall()

    counts = {}

    # --- gmiot_courses_learning (what_you_will_learn) ---
    col_l = client.get_or_create_collection(
        name="gmiot_courses_learning",
        metadata={"hnsw:space": "cosine"},
    )
    if col_l.count() > 0:
        print(f"  gmiot_courses_learning already populated ({col_l.count()} chunks) — skipping")
        counts["gmiot_courses_learning"] = col_l.count()
    else:
        valid = [(r, r["what_you_will_learn"]) for r in rows if r["what_you_will_learn"]]
        skipped = len(rows) - len(valid)
        print(f"  Embedding what_you_will_learn ({len(valid)} records"
              + (f", {skipped} skipped — NULL" if skipped else "") + ")...")

        ids       = [f"{r['course_id']}_learning" for r, _ in valid]
        texts     = [text for _, text in valid]
        metadatas = [
            {
                "course_id":         str(r["course_id"]),
                "course_name":       r["course_title"] or "",
                "field":             "what_you_will_learn",
                "ssa_category":      r["ssa_label"] or "",
                "provider":          r["provider"] or "",
                "level":             int(r["level"]) if r["level"] else 0,
                "qualification_type": r["qual_type"] or "",
            }
            for r, _ in valid
        ]

        embeddings = embed_in_batches(texts)
        col_l.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"  Done. {len(ids)} chunks added to gmiot_courses_learning")
        counts["gmiot_courses_learning"] = len(ids)

    # --- gmiot_courses_progression (progression) ---
    col_p = client.get_or_create_collection(
        name="gmiot_courses_progression",
        metadata={"hnsw:space": "cosine"},
    )
    if col_p.count() > 0:
        print(f"  gmiot_courses_progression already populated ({col_p.count()} chunks) — skipping")
        counts["gmiot_courses_progression"] = col_p.count()
    else:
        valid = [(r, r["progression"]) for r in rows if r["progression"]]
        skipped = len(rows) - len(valid)
        print(f"  Embedding progression ({len(valid)} records"
              + (f", {skipped} skipped — NULL" if skipped else "") + ")...")

        ids       = [f"{r['course_id']}_progression" for r, _ in valid]
        texts     = [text for _, text in valid]
        metadatas = [
            {
                "course_id":         str(r["course_id"]),
                "course_name":       r["course_title"] or "",
                "field":             "progression",
                "ssa_category":      r["ssa_label"] or "",
                "provider":          r["provider"] or "",
                "level":             int(r["level"]) if r["level"] else 0,
                "qualification_type": r["qual_type"] or "",
            }
            for r, _ in valid
        ]

        embeddings = embed_in_batches(texts)
        col_p.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"  Done. {len(ids)} chunks added to gmiot_courses_progression")
        counts["gmiot_courses_progression"] = len(ids)

    return counts


# ---------------------------------------------------------------------------
# Phase 2 — Job fields
# ---------------------------------------------------------------------------
def embed_job_fields(client: chromadb.PersistentClient) -> dict:
    print("\nPhase 2: Job field embeddings")

    conn = sqlite3.connect(JOBS_DB)
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute(
        "SELECT id, title, source, skills_required, typical_duties, entry_routes "
        "FROM jobs ORDER BY id"
    ).fetchall()

    counts = {}

    job_fields = [
        ("gmiot_jobs_skills", "skills_required",  "{id}_skills_only"),
        ("gmiot_jobs_duties", "typical_duties",   "{id}_duties"),
        ("gmiot_jobs_entry",  "entry_routes",      "{id}_entry"),
    ]

    for col_name, field, id_template in job_fields:
        col = client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )
        if col.count() > 0:
            print(f"  {col_name} already populated ({col.count()} chunks) — skipping")
            counts[col_name] = col.count()
            continue

        valid = [(r, r[field]) for r in all_rows if r[field]]
        skipped = len(all_rows) - len(valid)
        print(f"  Embedding {field} ({len(valid)} records"
              + (f", {skipped} skipped — NULL" if skipped else "") + ")...")

        ids       = [id_template.replace("{id}", str(r["id"])) for r, _ in valid]
        texts     = [text for _, text in valid]
        metadatas = [
            {
                "job_id": str(r["id"]),
                "title":  r["title"] or "",
                "field":  field,
                "source": r["source"] or "",
            }
            for r, _ in valid
        ]

        embeddings = embed_in_batches(texts)
        col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"  Done. {len(ids)} chunks added to {col_name}")
        counts[col_name] = len(ids)

    return counts


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(client: chromadb.PersistentClient) -> None:
    print("\nVerification:")
    collections = [
        ("gmiot_courses_learning",    "1_learning"),
        ("gmiot_courses_progression", "1_progression"),
        ("gmiot_jobs_skills",         "1_skills_only"),
        ("gmiot_jobs_duties",         "1_duties"),
        ("gmiot_jobs_entry",          "1_entry"),
    ]
    for col_name, spot_id in collections:
        try:
            col = client.get_collection(col_name)
            result = col.get(ids=[spot_id], include=["documents", "metadatas"])
            if result["documents"]:
                doc_preview = result["documents"][0][:80].replace("\n", " ")
                meta = result["metadatas"][0]
                print(f"  {col_name} ({col.count()} chunks)")
                print(f"    [{spot_id}] {doc_preview!r}")
                print(f"    metadata: {meta}")
            else:
                print(f"  {col_name} ({col.count()} chunks) — spot ID {spot_id!r} not found")
        except Exception as e:
            print(f"  {col_name} — ERROR: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    run_courses = "--jobs"   not in args
    run_jobs    = "--courses" not in args

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    all_counts = {}

    if run_courses:
        all_counts.update(embed_course_fields(client))

    if run_jobs:
        all_counts.update(embed_job_fields(client))

    print("\nSummary:")
    for name, count in all_counts.items():
        print(f"  {name:<32} {count:>5} chunks")

    verify(client)


if __name__ == "__main__":
    main()
