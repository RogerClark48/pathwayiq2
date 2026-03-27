import sqlite3

DB_PATH = r"C:\Dev\emiot-pathway-explorer\emiot_jobs_asset.db"

def show_enrichment_summary():
    conn = sqlite3.connect(DB_PATH)
    print("=== ENRICHMENT STATUS SUMMARY ===")
    rows = conn.execute("""
        SELECT enrichment_status, COUNT(*) as count
        FROM jobs
        GROUP BY enrichment_status
        ORDER BY count DESC
    """).fetchall()
    for status, count in rows:
        print(f"  {status or 'not started'}: {count}")
    conn.close()

def show_sample(status='ok', limit=2):
    conn = sqlite3.connect(DB_PATH)
    print(f"\n=== SAMPLE RECORDS: status='{status}' ===")
    rows = conn.execute("""
        SELECT title, source, enrichment_status, enriched_description
        FROM jobs
        WHERE enrichment_status = ?
        LIMIT ?
    """, (status, limit)).fetchall()
    for title, source, status, desc in rows:
        print(f"\n{'-'*50}")
        print(f"Title: {title} ({source})")
        print(f"Status: {status}")
        print(f"Description:\n{desc or 'NULL'}")
    conn.close()

def show_failed(limit=5):
    conn = sqlite3.connect(DB_PATH)
    print(f"\n=== FAILED RECORDS ===")
    rows = conn.execute("""
        SELECT title, source, url, enrichment_status
        FROM jobs
        WHERE enrichment_status IS NOT NULL
        AND enrichment_status != 'ok'
        LIMIT ?
    """, (limit,)).fetchall()
    for title, source, url, status in rows:
        print(f"\n  {title} ({source})")
        print(f"  Status: {status}")
        print(f"  URL: {url}")
    conn.close()

def clear_failed():
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("""
        SELECT COUNT(*) FROM jobs 
        WHERE enrichment_status != 'ok'
        AND enrichment_status IS NOT NULL
    """).fetchone()[0]
    conn.execute("""
        UPDATE jobs SET enriched_description = NULL, enrichment_status = NULL
        WHERE enrichment_status != 'ok'
        AND enrichment_status IS NOT NULL
    """)
    conn.commit()
    print(f"Cleared {count} failed records")
    conn.close()

def show_raw_fetch(title):
    """Show what we actually get back from fetching a specific job's URL"""
    import httpx
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT url FROM jobs WHERE title = ?
    """, (title,)).fetchone()
    conn.close()
    
    if not row:
        print(f"Job '{title}' not found")
        return
        
    url = row[0]
    print(f"Fetching: {url}")
    try:
        response = httpx.get(url, timeout=15, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PathwayIQ/1.0)"})
        print(f"Status: {response.status_code}")
        print(f"Content length: {len(response.text)}")
        print(f"\nFirst 2000 chars:")
        print(response.text[:2000])
    except Exception as e:
        print(f"Error: {e}")
def show_sample_by_source(source, limit=2):
    conn = sqlite3.connect(DB_PATH)
    print(f"\n=== SAMPLE RECORDS: source='{source}' ===")
    rows = conn.execute("""
        SELECT title, source, enrichment_status, enriched_description
        FROM jobs
        WHERE source = ?
        AND enrichment_status = 'ok'
        LIMIT ?
    """, (source, limit)).fetchall()
    for title, source, status, desc in rows:
        print(f"\n{'-'*50}")
        print(f"Title: {title} ({source})")
        print(f"Description:\n{desc or 'NULL'}")
    conn.close()

import chromadb

def show_chroma_summary():
    client = chromadb.PersistentClient(path=r"C:\Dev\pathwayiq\chroma_store")
    collection = client.get_collection("jobs")
    print(f"Chroma jobs collection: {collection.count()} records")

def show_by_status(status, limit=10):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT title, source, url
        FROM jobs
        WHERE enrichment_status = ?
        LIMIT ?
    """, (status, limit)).fetchall()
    print(f"\n=== STATUS: {status} ===")
    for title, source, url in rows:
        print(f"  {title} ({source})")
        print(f"  {url}")
    conn.close()
    
# ---- CHANGE WHAT RUNS HERE ----
show_chroma_summary()
show_sample_by_source('prospects', limit=2)
show_sample_by_source('ncs', limit=2)
show_failed(limit=10)
show_sample(status='ok', limit=1)
show_enrichment_summary()
show_by_status('insufficient_content', limit=10)
show_by_status('fetch_failed', limit=5)
# show_raw_fetch('Advertising media planner')
# clear_failed()