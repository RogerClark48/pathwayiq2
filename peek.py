import sqlite3

conn = sqlite3.connect(r'C:\Dev\emiot-pathway-explorer\job_roles_asset.db')
conn.execute("""
    UPDATE jobs SET enriched_description = NULL, enrichment_status = NULL
    WHERE enrichment_status != 'ok'
""")
conn.commit()
print("Cleared failed records")
conn.close()