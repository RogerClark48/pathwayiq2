"""
Pre-push script: fetches active access codes from the live Railway instance
and sets SEED_ACCESS_CODES so they survive redeployment.

Usage:
    python scripts/sync_access_codes.py [--dry-run] [--password SECRET]

Requires:
  - Railway CLI installed and authenticated
  - ADMIN_PASSWORD env var set, or pass --password on the command line
"""
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
import base64

RAILWAY_URL = "https://pathwayiq2-production-5c07.up.railway.app"

def main():
    dry_run = "--dry-run" in sys.argv

    password = os.environ.get("ADMIN_PASSWORD", "").strip()
    if not password and "--password" in sys.argv:
        idx = sys.argv.index("--password")
        if idx + 1 < len(sys.argv):
            password = sys.argv[idx + 1].strip()
    if not password:
        print("Provide password via --password SECRET or ADMIN_PASSWORD env var.")
        sys.exit(1)

    credentials = base64.b64encode(f"admin:{password}".encode()).decode()
    req = urllib.request.Request(
        f"{RAILWAY_URL}/admin/codes/export",
        headers={"Authorization": f"Basic {credentials}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            codes = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Request failed: HTTP {e.code} — wrong password?")
        sys.exit(1)
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(1)

    if not codes:
        print("No active access codes found on Railway instance.")
        sys.exit(0)

    print(f"Found {len(codes)} active code(s):")
    for c in codes:
        expiry = c["expires_at"] or "no expiry"
        print(f"  {c['code']}  ({c['label']})  expires: {expiry}")

    value = json.dumps(codes)

    if dry_run:
        print("\n[dry-run] Would set SEED_ACCESS_CODES to:")
        print(value)
        return

    try:
        result = subprocess.run(
            ["railway", "variables", "set", f"SEED_ACCESS_CODES={value}"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("\nSEED_ACCESS_CODES set on Railway successfully.")
        else:
            print(f"\nFailed to set Railway variable:\n{result.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        print("\nRailway CLI not found. Set this in the Railway dashboard manually:")
        print(f"\nSEED_ACCESS_CODES={value}")

if __name__ == "__main__":
    main()
