"""Reminder Engine Module for Multi-Stage Schedule and 10-Column Export."""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

from config.config import PRIMARY_REMINDER_BUFFER_DAYS, REMINDER_STAGES_DAYS
from refillcare.data.monthly_ingestion import determine_mobile_status


class RefillReminderEngine:
    """Manages multi-stage reminder schedules and delivery export formatting."""

    def __init__(self, primary_buffer_days: int = PRIMARY_REMINDER_BUFFER_DAYS):
        self.primary_buffer_days = primary_buffer_days

    def schedule_customer_reminders(
        self,
        customer_id: str,
        item_id: str,
        customer_name: str,
        item_name: str,
        phone_number: str,
        last_purchase_date: date,
        days_of_supply: float,
    ) -> List[Dict[str, Any]]:
        """Generate all multi-stage reminder milestones for a single purchase cycle."""
        expected_refill_date = last_purchase_date + timedelta(days=int(round(days_of_supply)))
        mob_status = determine_mobile_status(phone_number)

        milestones = []
        for offset in REMINDER_STAGES_DAYS:
            rem_date = expected_refill_date + timedelta(days=offset)
            stage_name = f"{abs(offset)}_days_before" if offset < 0 else (f"{offset}_days_after" if offset > 0 else "on_due_date")
            is_primary = (offset == -self.primary_buffer_days)

            milestones.append({
                "reminder_id": f"REM_{customer_id}_{item_id}_{rem_date.isoformat()}_{offset}",
                "customer_id": customer_id,
                "customer_name": customer_name,
                "phone_number": phone_number,
                "mobile_status": mob_status,
                "item_id": item_id,
                "item_name": item_name,
                "last_purchase_date": last_purchase_date.isoformat(),
                "estimated_days_of_supply": round(days_of_supply, 1),
                "expected_refill_date": expected_refill_date.isoformat(),
                "reminder_date": rem_date.isoformat(),
                "reminder_stage": stage_name,
                "is_primary_reminder": is_primary,
                "delivery_eligible": (mob_status == "Valid"),
            })

        return milestones

    @staticmethod
    def build_10_column_export_csv(reminders_df: pd.DataFrame) -> bytes:
        """Build exact 10-column delivery-ready CSV with only valid mobile numbers.

        Required Columns:
        1. customer_id
        2. customer_name
        3. phone_number
        4. mobile_status
        5. item_id
        6. medication_name
        7. last_purchase_date
        8. estimated_days_of_supply
        9. expected_refill_date
        10. reminder_date
        """
        exact_columns = [
            "customer_id",
            "customer_name",
            "phone_number",
            "mobile_status",
            "item_id",
            "medication_name",
            "last_purchase_date",
            "estimated_days_of_supply",
            "expected_refill_date",
            "reminder_date",
        ]

        if reminders_df.empty:
            empty_df = pd.DataFrame(columns=exact_columns)
            buf = io.StringIO()
            empty_df.to_csv(buf, index=False)
            return buf.getvalue().encode("utf-8")

        # Filter only valid mobile numbers for delivery
        df = reminders_df.copy()
        if "mobile_status" in df.columns:
            df = df[df["mobile_status"] == "Valid"]

        # Helper to safely extract series from dataframe
        def _get_series(source_df: pd.DataFrame, keys: list[str], default_val: str = "") -> pd.Series:
            for k in keys:
                if k in source_df.columns:
                    return source_df[k].astype(str)
            return pd.Series([default_val] * len(source_df), index=source_df.index, dtype=str)

        def _clean_phone(val: Any) -> str:
            if val is None or pd.isna(val):
                return ""
            s = str(val).strip()
            if s.endswith(".0"):
                s = s[:-2]
            s_digits = "".join(c for c in s if c.isdigit())
            return s_digits if s_digits else s

        def _clean_date(val: Any) -> str:
            if val is None or pd.isna(val):
                return "-"
            s = str(val).strip()
            if " " in s:
                s = s.split(" ")[0]
            if "T" in s:
                s = s.split("T")[0]
            return s

        raw_phone = _get_series(df, ["phone_number", "MOBILE_NO", "mobile_no"])
        cleaned_phones = raw_phone.apply(_clean_phone)

        export_df = pd.DataFrame(index=df.index)
        export_df["customer_id"] = _get_series(df, ["customer_id", "customerId"])
        export_df["customer_name"] = _get_series(df, ["customer_name", "customerName"], default_val="Unknown")
        export_df["phone_number"] = cleaned_phones
        export_df["mobile_status"] = "Valid"
        export_df["item_id"] = _get_series(df, ["item_id", "itemId"])
        export_df["medication_name"] = _get_series(df, ["medication_name", "item_name", "itemName", "Medication"], default_val="Unknown")
        export_df["last_purchase_date"] = _get_series(df, ["last_purchase_date", "invoice_date"]).apply(_clean_date)
        
        dos_series = None
        for k in ["estimated_days_of_supply", "Estimated Days of Supply", "predicted_days_until_refill", "predicted_interval_days", "dos_days"]:
            if k in df.columns:
                dos_series = pd.to_numeric(df[k], errors="coerce").fillna(30.0).round(1)
                break
        if dos_series is None:
            dos_series = pd.Series([30.0] * len(df), index=df.index)
        export_df["estimated_days_of_supply"] = dos_series

        export_df["expected_refill_date"] = _get_series(df, ["expected_refill_date", "Expected Refill Date"]).apply(_clean_date)
        export_df["reminder_date"] = _get_series(df, ["reminder_date", "Reminder Date", "target_send_date"]).apply(_clean_date)

        export_df = export_df[exact_columns].fillna("-")
        buf = io.StringIO()
        export_df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")
