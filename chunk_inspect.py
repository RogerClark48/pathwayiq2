import chromadb
import sqlite3
import json

CHROMA_PATH = "chroma_store"
JOBS_DB = "emiot_jobs_asset.db"
COURSES_DB = "emiot.sqlite"

client = chromadb.PersistentClient(path=CHROMA_PATH)
jobs_col = client.get_collection("jobs")
courses_col = client.get_collection("courses")

# ── 1. Collection overview ──────────────────────────────────────────
print("\n" + "="*60)
print("COLLECTION OVERVIEW")
print("="*60)
print(f"Jobs collection:    {jobs_col.count()} chunks")
print(f"Courses collection: {courses_col.count()} chunks")

# ── 2. List all unique chunk ID patterns ────────────────────────────
# Peek at first 20 IDs from each collection to understand ID structure
print("\n" + "="*60)
print("CHUNK ID PATTERNS (first 20 from each collection)")
print("="*60)

jobs_peek = jobs_col.peek(limit=20)
print("\nJobs IDs:")
for id_ in jobs_peek["ids"]:
    print(f"  {id_}")

courses_peek = courses_col.peek(limit=20)
print("\nCourses IDs:")
for id_ in courses_peek["ids"]:
    print(f"  {id_}")

# ── 3. All metadata keys present ────────────────────────────────────
print("\n" + "="*60)
print("METADATA KEYS PRESENT")
print("="*60)

jobs_meta_keys = set()
for m in jobs_peek["metadatas"]:
    jobs_meta_keys.update(m.keys())
print(f"\nJobs metadata keys:    {sorted(jobs_meta_keys)}")

courses_meta_keys = set()
for m in courses_peek["metadatas"]:
    courses_meta_keys.update(m.keys())
print(f"Courses metadata keys: {sorted(courses_meta_keys)}")

# ── 4. Full chunk content — 3 job samples ───────────────────────────
print("\n" + "="*60)
print("SAMPLE JOB CHUNKS (3 chunks, full content)")
print("="*60)

job_ids = jobs_peek["ids"][:3]
job_results = jobs_col.get(ids=job_ids, include=["documents", "metadatas"])

for i, (id_, doc, meta) in enumerate(zip(
    job_results["ids"],
    job_results["documents"],
    job_results["metadatas"]
)):
    print(f"\n--- Job chunk {i+1}: {id_} ---")
    print(f"Metadata: {json.dumps(meta, indent=2)}")
    print(f"Document text ({len(doc)} chars):")
    print(doc)
    print()

# ── 5. Full chunk content — all course chunk types ──────────────────
print("\n" + "="*60)
print("SAMPLE COURSE CHUNKS (one of each chunk type)")
print("="*60)

# Get all course IDs and group by type
all_courses = courses_col.get(include=["documents", "metadatas"])
seen_types = {}
for id_, doc, meta in zip(
    all_courses["ids"],
    all_courses["documents"],
    all_courses["metadatas"]
):
    # Infer chunk type from ID suffix
    chunk_type = id_.split("_")[-1] if "_" in id_ else "unknown"
    if chunk_type not in seen_types:
        seen_types[chunk_type] = (id_, doc, meta)

for chunk_type, (id_, doc, meta) in seen_types.items():
    print(f"\n--- Course chunk type: {chunk_type} | ID: {id_} ---")
    print(f"Metadata: {json.dumps(meta, indent=2)}")
    print(f"Document text ({len(doc)} chars):")
    print(doc)
    print()

# ── 6. Cross-collection retrieval sample ────────────────────────────
# Simulate what the LLM would actually receive:
# Pick one course, retrieve its top 5 career matches, show full text
print("\n" + "="*60)
print("SIMULATED LLM INPUT — cross-collection retrieval")
print("What the LLM would see for one course → careers query")
print("="*60)

# Pick first course that has an overview or skills chunk
source_id = None
source_doc = None
source_meta = None
for id_, doc, meta in zip(
    all_courses["ids"],
    all_courses["documents"],
    all_courses["metadatas"]
):
    if "overview" in id_ or "skills" in id_:
        source_id = id_
        source_doc = doc
        source_meta = meta
        break

if source_id:
    # Get its embedding
    vec_result = courses_col.get(ids=[source_id], include=["embeddings"])
    vector = vec_result["embeddings"][0]

    # Query jobs collection
    matches = jobs_col.query(
        query_embeddings=[vector],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nSource chunk: {source_id}")
    print(f"Source metadata: {json.dumps(source_meta, indent=2)}")
    print(f"Source text: {source_doc[:300]}{'...' if len(source_doc) > 300 else ''}")
    print(f"\nTop 5 matched job chunks:")

    for i, (id_, doc, meta, dist) in enumerate(zip(
        matches["ids"][0],
        matches["documents"][0],
        matches["metadatas"][0],
        matches["distances"][0]
    )):
        score = round((1 - dist) * 100, 1)
        print(f"\n  Match {i+1} ({score}%): {id_}")
        print(f"  Metadata: {json.dumps(meta, indent=2)}")
        print(f"  Text ({len(doc)} chars): {doc[:200]}{'...' if len(doc) > 200 else ''}")
else:
    print("No overview/skills chunk found — check ID naming convention above.")

# ── 7. SQLite fields available for augmentation ─────────────────────
print("\n" + "="*60)
print("SQLITE FIELDS AVAILABLE FOR LLM AUGMENTATION")
print("(fields in DB that are NOT in Chroma metadata)")
print("="*60)

conn_jobs = sqlite3.connect(JOBS_DB)
jobs_cols = [r[1] for r in conn_jobs.execute("PRAGMA table_info(jobs)").fetchall()]
print(f"\nJobs DB columns: {jobs_cols}")
conn_jobs.close()

conn_courses = sqlite3.connect(COURSES_DB)
# Find the courses table name
tables = [r[0] for r in conn_courses.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()]
print(f"\nCourses DB tables: {tables}")
for table in tables:
    cols = [r[1] for r in conn_courses.execute(f"PRAGMA table_info({table})").fetchall()]
    print(f"  {table} columns: {cols}")
conn_courses.close()

print("\n" + "="*60)
print("DONE — review output above before designing LLM prompt")
print("="*60)
