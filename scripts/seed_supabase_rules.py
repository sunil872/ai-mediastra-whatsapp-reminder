"""Seeding script to populate Supabase medication_classification_rules table.

Transfers the hardcoded baseline rules from medication_classifier.py
into the Supabase PostgreSQL database for dynamic live management.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT.parent / ".env")

from refillcare.data.medication_classifier import (
    EXCLUDED_CATEGORIES,
    CHRONIC_STRONG_MARKERS,
)


def seed_supabase_medication_rules():
    """Seed baseline exclusion and chronic markers into Supabase."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        return

    try:
        from supabase import create_client, Client
        supabase: Client = create_client(supabase_url, supabase_key)
    except ImportError:
        print("❌ Error: 'supabase' package is not installed. Please run: pip install supabase")
        return
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        return

    print(f"[+] Connected to Supabase: {supabase_url}")

    records = []
    
    # 1. Add all hardcoded exclusion keywords as EXCLUDE rules
    for category, keywords in EXCLUDED_CATEGORIES.items():
        cat_clean = category.replace("_", " ").title()
        for kw in keywords:
            kw_clean = kw.strip().upper()
            if kw_clean:
                records.append({
                    "medication_name_or_keyword": kw_clean,
                    "rule_type": "EXCLUDE",
                    "notes": f"System baseline ({cat_clean})",
                    "is_active": True,
                })

    # 2. Add all hardcoded chronic molecules as CHRONIC rules
    for marker in CHRONIC_STRONG_MARKERS:
        marker_clean = marker.strip().upper()
        if marker_clean:
            records.append({
                "medication_name_or_keyword": marker_clean,
                "rule_type": "CHRONIC",
                "notes": "System baseline chronic therapy",
                "is_active": True,
            })

    # Remove duplicates preserving order
    unique_map = {}
    for r in records:
        key = (r["medication_name_or_keyword"], r["rule_type"])
        unique_map[key] = r
    unique_records = list(unique_map.values())

    print(f"[+] Total unique rules to seed: {len(unique_records)} (Exclusions: {sum(1 for r in unique_records if r['rule_type']=='EXCLUDE')}, Chronic: {sum(1 for r in unique_records if r['rule_type']=='CHRONIC')})")

    # Insert in batches of 100
    batch_size = 100
    inserted_count = 0
    for i in range(0, len(unique_records), batch_size):
        batch = unique_records[i:i + batch_size]
        try:
            res = supabase.table("medication_classification_rules").upsert(
                batch,
                on_conflict="medication_name_or_keyword,rule_type"
            ).execute()
            inserted_count += len(batch)
            print(f"  * Uploaded batch {i // batch_size + 1} ({inserted_count}/{len(unique_records)})")
        except Exception as e:
            print(f"  ! Warning on batch {i // batch_size + 1}: {e}")

    print(f"[SUCCESS] Successfully seeded {inserted_count} medication classification rules to Supabase!")


if __name__ == "__main__":
    seed_supabase_medication_rules()
