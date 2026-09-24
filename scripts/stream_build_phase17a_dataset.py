"""Fast streaming build of Phase 17A dataset using openpyxl read-only streaming."""

from __future__ import annotations

import sys
import time
import csv
from pathlib import Path
from datetime import datetime, date
import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "refillcare" / "customer_data_fields.csv"
DATASET_FOLDER = Path(r"D:\Downloads\yard_global\Reminder Dataset")

EXCEL_FILES = [
    "24122020 TO 31032021 A12.xlsx",
    "01042021 TO 31032022 A12.xlsx",
    "01042022 TO 31032023 A12.xlsx",
    "01042023 TO 31032024 A12.xlsx",
    "01042024 TO 31032025 A12.xlsx",
    "01042025 TO 30092025 A12.xlsx",
    "01102025 TO 31032026 A12.xlsx",
    "01042026 TO 30062026 A12.xlsx",
    "01072026 TO 31082026 A12.xlsx",
]

TARGET_HEADER = [
    "invoice_date",
    "invoice_number",
    "customerName",
    "itemCode",
    "itemName",
    "packing",
    "batchNo",
    "mfgDate",
    "expiryDate",
    "quantity",
    "freeQuantity",
    "rate",
    "saleRate",
    "mrp",
    "invoice_total",
    "discountPercent",
    "item_discount",
    "gstAmount",
    "netAmount",
    "costPrice",
    "MOBILE_NO",
    "customerId",
    "itemId",
]


def parse_cell_date(val) -> str:
    """Parse cell value into DD/MM/YYYY string."""
    if val is None:
        return ""
    if isinstance(val, (datetime, date)):
        return val.strftime("%d/%m/%Y")
    s = str(val).strip()
    if not s or "total" in s.lower():
        return ""
    # Try parsing string formats
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            pass
    # Try dateutil / pandas if needed
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        if pd.notna(dt):
            return dt.strftime("%d/%m/%Y")
    except Exception:
        pass
    return s


def clean_str(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s in ("nan", "None", "NaN", "null", "NULL"):
        return ""
    return s


def clean_phone(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s in ("nan", "None", "NaN", "null", "NULL"):
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_streaming_dataset() -> None:
    start_time = time.time()
    print("=" * 70)
    print("STREAMING REFILLCARE SOURCE DATASET BUILD (PHASE 17A)")
    print("=" * 70)
    print(f"Source folder: {DATASET_FOLDER}")
    print(f"Target file:   {OUTPUT_CSV}\n")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temp_output = OUTPUT_CSV.with_suffix(".tmp")

    total_written_rows = 0
    file_stats = []

    with open(temp_output, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(TARGET_HEADER)

        for idx, fname in enumerate(EXCEL_FILES, 1):
            fpath = DATASET_FOLDER / fname
            t0 = time.time()
            print(f"[{idx}/9] Streaming {fname}...", flush=True)

            wb = openpyxl.load_workbook(fpath, read_only=True, data_only=True)
            sheet = wb["Sale Book With Item Details"]

            file_rows = 0
            first_date = None
            last_date = None

            # Iterate rows starting from row 7 (row 6 is header, rows 1-5 are title)
            for row_vals in sheet.iter_rows(min_row=7, values_only=True):
                if not row_vals or row_vals[0] is None:
                    continue
                raw_date = row_vals[0]
                if "total" in str(raw_date).lower():
                    continue

                formatted_date = parse_cell_date(raw_date)
                if not formatted_date:
                    continue

                if first_date is None:
                    first_date = formatted_date
                last_date = formatted_date

                trn_no = clean_str(row_vals[1])
                party_name = clean_str(row_vals[2])
                item_code = clean_str(row_vals[3])
                item_name = clean_str(row_vals[4])
                pack = clean_str(row_vals[5])
                batch = clean_str(row_vals[6])
                mfg_dt = clean_str(row_vals[7])
                expiry = clean_str(row_vals[8])
                qty = clean_str(row_vals[9])
                free = clean_str(row_vals[10])
                bill_rate = clean_str(row_vals[11])
                sale_rate = clean_str(row_vals[12])
                mrp = clean_str(row_vals[14])
                inv_amt = clean_str(row_vals[27])
                dis_pct = clean_str(row_vals[16])
                dis_amt = clean_str(row_vals[17])
                gst_amt = clean_str(row_vals[24])
                net_amt = clean_str(row_vals[25])
                cost = clean_str(row_vals[26])
                mobile = clean_phone(row_vals[29])

                # Identity mappings
                customer_id = party_name  # Derived directly from party name, NEVER phone
                item_id = item_code      # Preserved directly from item code

                out_row = [
                    formatted_date,
                    trn_no,
                    party_name,
                    item_code,
                    item_name,
                    pack,
                    batch,
                    mfg_dt,
                    expiry,
                    qty,
                    free,
                    bill_rate,
                    sale_rate,
                    mrp,
                    inv_amt,
                    dis_pct,
                    dis_amt,
                    gst_amt,
                    net_amt,
                    cost,
                    mobile,
                    customer_id,
                    item_id,
                ]

                writer.writerow(out_row)
                file_rows += 1

            wb.close()
            total_written_rows += file_rows
            elapsed = time.time() - t0
            file_stats.append({
                "file": fname,
                "rows": file_rows,
                "first_date": first_date,
                "last_date": last_date,
                "time_sec": elapsed,
            })
            print(f"      Written: {file_rows:,} rows in {elapsed:.1f}s | Dates: {first_date} -> {last_date}", flush=True)

    # Atomic rename
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
    temp_output.rename(OUTPUT_CSV)

    file_size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f"SUCCESS: Generated {OUTPUT_CSV} ({file_size_mb:.2f} MB)")
    print(f"Total rows written: {total_written_rows:,}")
    print(f"Total time elapsed: {time.time() - start_time:.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    build_streaming_dataset()
