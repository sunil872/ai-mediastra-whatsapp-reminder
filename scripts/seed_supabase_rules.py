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
    CHRONIC_SALT_MOLECULES,
    ACUTE_EXCLUDED_SALT_MOLECULES,
    extract_salt_molecules,
    list_classification_rules,
)


def seed_supabase_medication_rules():
    """Seed baseline exclusion, chronic markers, chemical salts, and catalog brands into Supabase."""
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

    # 2. Add all acute/excluded salt molecules
    for molecule in ACUTE_EXCLUDED_SALT_MOLECULES:
        m_clean = molecule.strip().upper()
        if m_clean:
            records.append({
                "medication_name_or_keyword": m_clean,
                "rule_type": "EXCLUDE",
                "notes": "Acute / PRN / Excluded chemical salt molecule",
                "is_active": True,
            })

    # 3. Add all hardcoded chronic molecules as CHRONIC rules
    for marker in CHRONIC_STRONG_MARKERS:
        marker_clean = marker.strip().upper()
        if marker_clean:
            records.append({
                "medication_name_or_keyword": marker_clean,
                "rule_type": "CHRONIC",
                "notes": "System baseline chronic therapy",
                "is_active": True,
            })

    # 4. Add all chronic salt molecules
    for molecule in CHRONIC_SALT_MOLECULES:
        m_clean = molecule.strip().upper()
        if m_clean:
            records.append({
                "medication_name_or_keyword": m_clean,
                "rule_type": "CHRONIC",
                "notes": "Chronic maintenance chemical salt molecule",
                "is_active": True,
            })

    # 5. Mine leading oral chronic brands from SALT WISE ITEMS.csv catalog
    import csv
    import re
    catalog_path = PROJECT_ROOT / "data" / "refillcare" / "SALT WISE ITEMS.csv"
    if catalog_path.exists():
        print(f"[+] Mining chronic brands from {catalog_path.name}...")
        chronic_brand_counts = {}
        with open(catalog_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("S.No,"):
                    header = [c.strip() for c in line.split(",")]
                    break
            reader = csv.DictReader(f, fieldnames=header)
            for row in reader:
                cat = row.get("CATEGORY", "").strip().upper()
                salt = row.get("SALT", "").strip().upper()
                name = row.get("Item Name", "").strip().upper()
                pack = row.get("PACK", "").strip().upper()

                if cat in {
                    "SYRUPS", "EYE/EAR DRPOS", "ORAL DROPS", "OINTMENTS/CREAMS/GEL/PASTE",
                    "LOTIONS/SHAMPOOS", "SOAPS", "GEL", "SURGICAL", "CONTAINERS", "NASAL SPRAY",
                    "SACHETS", "POWDERS", "GENERAL"
                }:
                    continue
                if re.search(r"\b(?:SYP|SYRUP|SUSP|SUSPENSION|DROP|DROPS|DRPOS|OINT|CREAM|CRM|GEL|LOTION|SOAP|SHAMPOO|WASH|SPRAY|INJ|SACHET|GRANULES)\b", name):
                    continue

                mols = extract_salt_molecules(salt)
                if any(m in ACUTE_EXCLUDED_SALT_MOLECULES for m in mols):
                    continue
                if any(m in CHRONIC_SALT_MOLECULES for m in mols):
                    parts = name.split()
                    if parts:
                        brand = re.split(r"[-_\d]", parts[0].strip())[0].strip()
                        if len(brand) >= 3 and not re.search(r"^(TAB|CAP|INJ|SYP|KIT)$", brand):
                            chronic_brand_counts[brand] = chronic_brand_counts.get(brand, 0) + 1

        # Add brands with frequency >= 2
        for brand, count in chronic_brand_counts.items():
            if count >= 2:
                records.append({
                    "medication_name_or_keyword": brand,
                    "rule_type": "CHRONIC",
                    "notes": f"Chronic maintenance brand catalog ({count} items in store)",
                    "is_active": True,
                })
        print(f"    * Added {len([b for b, c in chronic_brand_counts.items() if c >= 2])} catalog chronic brands.")

    # Remove duplicates preserving order
    unique_map = {}
    for r in records:
        key = (r["medication_name_or_keyword"], r["rule_type"])
        unique_map[key] = r
    unique_records = list(unique_map.values())

    print(f"[+] Total unique rules to sync: {len(unique_records)} (Exclusions: {sum(1 for r in unique_records if r['rule_type']=='EXCLUDE')}, Chronic: {sum(1 for r in unique_records if r['rule_type']=='CHRONIC')})")

    # 6. Correct Eye Drops in Supabase: Convert previous CHRONIC eye drops to EXCLUDE
    eye_drop_corrections = [
        "REFRESH TEARS", "TIMOLOL", "CARBOXYMETHYLCELLULOSE", "COMBIGAN", "DORZOX",
        "SYSTANE", "LUBREX", "BIMATOPROST", "LATANOPROST", "TRAVAPROST", "BRIMONIDINE", "ALPHAGAN"
    ]
    print("[+] Enforcing eye drops & drops exclusion rule in Supabase...")
    for ed in eye_drop_corrections:
        try:
            supabase.table("medication_classification_rules").delete().eq("medication_name_or_keyword", ed).eq("rule_type", "CHRONIC").execute()
            supabase.table("medication_classification_rules").upsert({
                "medication_name_or_keyword": ed,
                "rule_type": "EXCLUDE",
                "notes": "Ophthalmic formulation exclusion (Eye Drop)",
                "is_active": True,
            }, on_conflict="medication_name_or_keyword,rule_type").execute()
        except Exception as e:
            pass

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

    # 7. Auto-Export Complete Up-to-Date Rules to exports/
    print("[+] Syncing complete updated rules to exports/medication_classification_rules_complete.csv...")
    try:
        import pandas as pd
        all_rules = list_classification_rules(limit=None)
        if all_rules:
            export_dir = PROJECT_ROOT / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / "medication_classification_rules_complete.csv"
            df_rules = pd.DataFrame(all_rules)
            df_rules.to_csv(export_path, index=False)
            print(f"    * Exported {len(df_rules):,} complete rules to {export_path}")
    except Exception as e:
        print(f"    ! Warning exporting rules: {e}")


if __name__ == "__main__":
    seed_supabase_medication_rules()
