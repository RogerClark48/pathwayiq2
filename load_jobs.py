import sqlite3
import chromadb
import ollama

# Connect to the jobs database
DB_PATH = r"C:\Dev\emiot-pathway-explorer\job_roles_asset.db"
CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"

print("Connecting to jobs database...")
conn = sqlite3.connect(DB_PATH)

# Pull only original source fields - no ESCO preprocessing columns
rows = conn.execute("""
    SELECT id, title, url, source, description, 
           entry_requirements, progression_routes
    FROM jobs
    WHERE is_active = 1
    AND description IS NOT NULL
    
""").fetchall()

conn.close()
print(f"Loaded {len(rows)} jobs from database")

# Set up persistent Chroma store
print("Setting up Chroma vector store...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    chroma_client.delete_collection("jobs")
    print("Deleted old jobs collection")
except:
    pass
jobs_collection = chroma_client.get_or_create_collection(
    name="jobs",
    metadata={"hnsw:space": "cosine"})
# Embed and store each job
print("Embedding jobs - this may take a minute...")
for i, row in enumerate(rows):
    job_id, title, url, source, description, entry_req, progression = row
    
    # Combine the richest text fields for embedding
    text_to_embed = f"{title}\n\n{description or ''}\n\n{entry_req or ''}\n\n{progression or ''}"
    text_to_embed = text_to_embed.strip()
    
    # Generate embedding
    response = ollama.embeddings(model="nomic-embed-text", prompt=text_to_embed)
    embedding = response["embedding"]
    
    # Store in Chroma with metadata
    jobs_collection.add(
        ids=[str(job_id)],
        embeddings=[embedding],
        documents=[text_to_embed],
        metadatas=[{
            "title": title or "",
            "url": url or "",
            "source": source or ""
        }]
    )
    
    print(f"  [{i+1}/{len(rows)}] {title}")

print(f"\nDone. {len(rows)} jobs stored in Chroma.")
print(f"Store location: {CHROMA_PATH}")