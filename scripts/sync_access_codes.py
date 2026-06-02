"""
Pre-push script: reads active access codes from analytics.db and sets
SEED_ACCESS_CODES on Railway so they survive redeployment.

Usage:
    python scripts/sync_access_codes.py [--dry-run]

Requires Railway CLI to be installed and authenticated.
Only exports codes that are not yet expired.
"""
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ANALYTICS_DB = Path(__file__).parent.parent / "analytics.db"

def main():
    dry_run = "--dry-run" in sys.argv

    if not ANALYTICS_DB.exists():
        print(f"analytics.db not found at {ANALYTICS_DB}")
        sys.exit(1)

    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc).isoformat()

    rows = conn.execute(
        "SELECT code, label, expires_at FROM access_codes "
        "WHERE expires_at IS NULL OR expires_at > ?",
        (now,)
    ).fetchall()
    conn.close()

    if not rows:
        print("No active access codes found in analytics.db.")
        sys.exit(0)

    codes = [
        {"code": r["code"], "label": r["label"], "expires_at": r["expires_at"]}
        for r in rows
    ]
    value = json.dumps(codes)

    print(f"Found {len(codes)} active code(s):")
    for c in codes:
        expiry = c["expires_at"] or "no expiry"
        print(f"  {c['code']}  ({c['label']})  expires: {expiry}")

    if dry_run:
        print("\n[dry-run] Would set SEED_ACCESS_CODES to:")
        print(value)
        return

    result = subprocess.run(
        ["railway", "variables", "set", f"SEED_ACCESS_CODES={value}"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("\nSEED_ACCESS_CODES set on Railway successfully.")
    else:
        print(f"\nFailed to set Railway variable:\n{result.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
