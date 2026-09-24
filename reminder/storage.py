"""RefillCare Persistent Reminder State & Audit Storage."""

from __future__ import annotations

import os
import sqlite3
import logging
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING
from contextlib import contextmanager
import pandas as pd

if TYPE_CHECKING:
    from reminder.dispatch import DispatchResult

logger = logging.getLogger("refillcare.storage")

DEFAULT_DB_PATH = "data/refillcare/processed/refillcare.db"


def _extract_phone_last4(raw_phone: Optional[str]) -> str:
    """Safely extract the last 4 digits of a phone number for audit logs."""
    if not raw_phone:
        return ""
    digits = "".join(c for c in str(raw_phone) if c.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits


def _get_utc_now_iso() -> str:
    """Return ISO format string of current UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RefillCareStorage:
    """Persistent SQLite database manager for RefillCare reminder audit records."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initialize SQLite storage and ensure tables exist."""
        if db_path is not None:
            self.db_path = Path(db_path).resolve()
        else:
            env_path = os.getenv("REFILLCARE_DB_PATH")
            if env_path:
                self.db_path = Path(env_path).resolve()
            else:
                self.db_path = Path(DEFAULT_DB_PATH).resolve()

        # Ensure parent directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Create a sqlite3 connection context manager that guarantees closing."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Create reminder_audit table and indices if they do not exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS reminder_audit (
            reminder_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            customer_name TEXT,
            item_name TEXT,
            phone_last4 TEXT,
            expected_refill_date TEXT,
            reminder_date TEXT,
            reminder_stage TEXT,
            history_quality TEXT,
            status TEXT NOT NULL,
            attempt_count INTEGER DEFAULT 0,
            provider_message_id TEXT,
            error_category TEXT,
            error_message TEXT,
            is_dry_run INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_attempt_at TEXT
        );
        """
        create_snapshots_table_sql = """
        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            customer_name TEXT,
            item_name TEXT,
            last_purchase_date TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            estimated_days_of_supply REAL,
            expected_refill_date TEXT,
            reminder_date TEXT,
            prediction_source TEXT,
            status TEXT DEFAULT 'pending',
            actual_next_purchase_date TEXT,
            outcome_error_days REAL,
            created_at TEXT NOT NULL,
            import_batch_id TEXT
        );
        """
        create_batches_table_sql = """
        CREATE TABLE IF NOT EXISTS import_batches (
            import_batch_id TEXT PRIMARY KEY,
            upload_timestamp TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            detected_date_range TEXT,
            record_count INTEGER NOT NULL,
            records_inserted INTEGER NOT NULL,
            records_skipped INTEGER NOT NULL,
            records_requiring_review INTEGER NOT NULL,
            processing_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_active INTEGER DEFAULT 1
        );
        """
        create_indices_sql = [
            "CREATE INDEX IF NOT EXISTS idx_audit_status ON reminder_audit(status);",
            "CREATE INDEX IF NOT EXISTS idx_audit_customer ON reminder_audit(customer_id);",
            "CREATE INDEX IF NOT EXISTS idx_audit_reminder_date ON reminder_audit(reminder_date);",
            "CREATE INDEX IF NOT EXISTS idx_audit_updated_at ON reminder_audit(updated_at);",
            "CREATE INDEX IF NOT EXISTS idx_snap_cust_item ON prediction_snapshots(customer_id, item_id);",
            "CREATE INDEX IF NOT EXISTS idx_snap_pred_date ON prediction_snapshots(prediction_date);",
            "CREATE INDEX IF NOT EXISTS idx_snap_exp_date ON prediction_snapshots(expected_refill_date);",
            "CREATE INDEX IF NOT EXISTS idx_snap_status ON prediction_snapshots(status);",
            "CREATE INDEX IF NOT EXISTS idx_snap_batch ON prediction_snapshots(import_batch_id);",
            "CREATE INDEX IF NOT EXISTS idx_batch_status ON import_batches(is_active);",
        ]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            cursor.execute(create_snapshots_table_sql)
            cursor.execute(create_batches_table_sql)
            # Safe migration: add import_batch_id to prediction_snapshots if missing
            try:
                cursor.execute("ALTER TABLE prediction_snapshots ADD COLUMN import_batch_id TEXT;")
            except sqlite3.OperationalError:
                pass  # Column already exists
            for idx_sql in create_indices_sql:
                cursor.execute(idx_sql)
            conn.commit()

    def insert_or_update_scheduled_reminder(self, reminder_data: Dict[str, Any]) -> None:
        """Insert a scheduled reminder record if it does not already exist."""
        rem_id = str(reminder_data.get("reminder_id", "")).strip()
        if not rem_id:
            return

        cid = str(reminder_data.get("customerId", reminder_data.get("customer_id", ""))).strip()
        iid = str(reminder_data.get("itemId", reminder_data.get("item_id", ""))).strip()
        cname = str(reminder_data.get("customerName", reminder_data.get("customer_name", cid)))
        iname = str(reminder_data.get("itemName", reminder_data.get("item_name", iid)))
        phone_raw = str(reminder_data.get("MOBILE_NO", reminder_data.get("phone1", reminder_data.get("phone", ""))))
        phone_l4 = _extract_phone_last4(phone_raw)
        exp_date = str(reminder_data.get("expected_refill_date", ""))
        rem_date = str(reminder_data.get("reminder_date", ""))
        stage_val = str(reminder_data.get("reminder_stage", ""))
        hist_qual = str(reminder_data.get("history_quality", "medium_history"))
        status = str(reminder_data.get("status", "scheduled"))
        now_str = _get_utc_now_iso()

        sql = """
        INSERT INTO reminder_audit (
            reminder_id, customer_id, item_id, customer_name, item_name,
            phone_last4, expected_refill_date, reminder_date, reminder_stage,
            history_quality, status, attempt_count, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        ON CONFLICT(reminder_id) DO UPDATE SET
            customer_name = excluded.customer_name,
            item_name = excluded.item_name,
            phone_last4 = excluded.phone_last4,
            history_quality = excluded.history_quality,
            updated_at = excluded.updated_at
        WHERE reminder_audit.status NOT IN ('accepted', 'cancelled');
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                sql,
                (rem_id, cid, iid, cname, iname, phone_l4, exp_date, rem_date, stage_val, hist_qual, status, now_str, now_str)
            )
            conn.commit()

    def bulk_insert_or_update_scheduled_reminders(
        self,
        reminders: Union[pd.DataFrame, List[Dict[str, Any]]],
    ) -> None:
        """Batch insert or update scheduled reminders in a single fast SQLite transaction."""
        if isinstance(reminders, pd.DataFrame):
            records = reminders.to_dict("records")
        elif isinstance(reminders, (list, tuple)):
            records = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in reminders]
        else:
            return

        if not records:
            return

        now_str = _get_utc_now_iso()
        tuples_to_insert = []
        for r in records:
            rem_id = str(r.get("reminder_id", "")).strip()
            if not rem_id:
                continue
            cid = str(r.get("customerId", r.get("customer_id", ""))).strip()
            iid = str(r.get("itemId", r.get("item_id", ""))).strip()
            cname = str(r.get("customerName", r.get("customer_name", r.get("Customer", cid))))
            iname = str(r.get("itemName", r.get("item_name", r.get("Medicine", iid))))
            phone_raw = str(r.get("MOBILE_NO", r.get("phone1", r.get("phone", r.get("Delivery Phone", "")))))
            phone_l4 = _extract_phone_last4(phone_raw)
            exp_date = str(r.get("expected_refill_date", r.get("Expected Refill Date", "")))
            rem_date = str(r.get("reminder_date", r.get("Reminder Date", "")))
            stage_val = str(r.get("reminder_stage", r.get("Reminder Stage", "")))
            hist_qual = str(r.get("history_quality", r.get("History Quality", "medium_history")))
            status = str(r.get("status", r.get("Status", "scheduled"))).lower()

            tuples_to_insert.append((
                rem_id, cid, iid, cname, iname, phone_l4, exp_date, rem_date, stage_val, hist_qual, status, now_str, now_str
            ))

        if not tuples_to_insert:
            return

        sql = """
        INSERT INTO reminder_audit (
            reminder_id, customer_id, item_id, customer_name, item_name,
            phone_last4, expected_refill_date, reminder_date, reminder_stage,
            history_quality, status, attempt_count, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        ON CONFLICT(reminder_id) DO UPDATE SET
            customer_name = excluded.customer_name,
            item_name = excluded.item_name,
            phone_last4 = excluded.phone_last4,
            history_quality = excluded.history_quality,
            updated_at = excluded.updated_at
        WHERE reminder_audit.status NOT IN ('accepted', 'cancelled');
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, tuples_to_insert)
            conn.commit()

    def record_dispatch_start(
        self,
        reminder_data: Dict[str, Any],
        is_dry_run: bool = False,
    ) -> int:
        """Mark a reminder as sending and increment attempt_count.

        Returns:
            new_attempt_count (int)
        """
        rem_id = str(reminder_data.get("reminder_id", "")).strip()
        now_str = _get_utc_now_iso()
        cid = str(reminder_data.get("customerId", reminder_data.get("customer_id", ""))).strip()
        iid = str(reminder_data.get("itemId", reminder_data.get("item_id", ""))).strip()
        cname = str(reminder_data.get("customerName", reminder_data.get("customer_name", cid)))
        iname = str(reminder_data.get("itemName", reminder_data.get("item_name", iid)))
        phone_l4 = _extract_phone_last4(reminder_data.get("MOBILE_NO", reminder_data.get("phone1", reminder_data.get("phone", ""))))
        exp_date = str(reminder_data.get("expected_refill_date", ""))
        rem_date = str(reminder_data.get("reminder_date", ""))
        stage_val = str(reminder_data.get("reminder_stage", ""))
        hist_qual = str(reminder_data.get("history_quality", "medium_history"))

        sql = """
        INSERT INTO reminder_audit (
            reminder_id, customer_id, item_id, customer_name, item_name,
            phone_last4, expected_refill_date, reminder_date, reminder_stage,
            history_quality, status, attempt_count, is_dry_run, created_at, updated_at, last_attempt_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'sending', 1, ?, ?, ?, ?)
        ON CONFLICT(reminder_id) DO UPDATE SET
            status = 'sending',
            attempt_count = reminder_audit.attempt_count + 1,
            is_dry_run = excluded.is_dry_run,
            updated_at = excluded.updated_at,
            last_attempt_at = excluded.last_attempt_at;
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                sql,
                (rem_id, cid, iid, cname, iname, phone_l4, exp_date, rem_date, stage_val, hist_qual, 1 if is_dry_run else 0, now_str, now_str, now_str)
            )
            conn.commit()

        # Retrieve new attempt count
        rec = self.get_reminder(rem_id)
        return int(rec["attempt_count"]) if rec else 1

    def record_dispatch_result(
        self,
        result: DispatchResult,
        is_dry_run: bool = False,
    ) -> None:
        """Update audit record with the final dispatch outcome."""
        now_str = _get_utc_now_iso()
        rem_id = result.reminder_id
        final_status = result.status
        prov_id = result.provider_message_id or ""
        err_cat = result.error_category or ""
        err_msg = result.message or ""
        is_dry_val = 1 if (is_dry_run or result.dry_run) else 0

        # Also populate other fields if inserting for the first time
        cid = result.customerId
        iid = result.itemId
        cname = result.customerName
        phone_l4 = _extract_phone_last4(result.phone_masked)
        exp_date = result.expected_refill_date
        stage_val = result.reminder_stage

        sql = """
        INSERT INTO reminder_audit (
            reminder_id, customer_id, item_id, customer_name, item_name,
            phone_last4, expected_refill_date, reminder_date, reminder_stage,
            history_quality, status, attempt_count, provider_message_id,
            error_category, error_message, is_dry_run, created_at, updated_at, last_attempt_at
        ) VALUES (?, ?, ?, ?, '', ?, ?, '', ?, '', ?, 1, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(reminder_id) DO UPDATE SET
            status = excluded.status,
            provider_message_id = CASE WHEN excluded.provider_message_id != '' THEN excluded.provider_message_id ELSE reminder_audit.provider_message_id END,
            error_category = excluded.error_category,
            error_message = excluded.error_message,
            is_dry_run = excluded.is_dry_run,
            updated_at = excluded.updated_at;
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                sql,
                (rem_id, cid, iid, cname, phone_l4, exp_date, stage_val, final_status, prov_id, err_cat, err_msg, is_dry_val, now_str, now_str, now_str)
            )
            conn.commit()

    def get_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single reminder record by reminder_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reminder_audit WHERE reminder_id = ?", (reminder_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def is_send_allowed(self, reminder_id: str) -> Tuple[bool, str]:
        """Check if sending is permitted for a given reminder ID.

        Returns:
            (allowed: bool, reason: str)
        """
        rec = self.get_reminder(reminder_id)
        if not rec:
            return True, "No prior dispatch record. Ready for dispatch."

        status = rec.get("status", "scheduled")
        is_dry = bool(rec.get("is_dry_run", 0))

        # Real accepted messages must NEVER be resent
        if status == "accepted" and not is_dry:
            return False, "Duplicate Prevention: Reminder was already accepted by WhatsApp provider."

        # Cancelled messages must not be sent
        if status == "cancelled":
            return False, "Cycle Status: Reminder has been cancelled."

        # Currently in-flight dispatch
        if status == "sending":
            return False, "Concurrent Safety: Reminder is currently being processed."

        # Failed or scheduled records can be dispatched/retried
        if status == "failed":
            return True, f"Eligible for retry (Previous attempt #{rec.get('attempt_count', 1)} failed)."

        return True, "Eligible for dispatch."

    def get_all_reminders(
        self,
        status: Optional[str] = None,
        customer_id: Optional[str] = None,
        reminder_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve audit records matching optional query filters."""
        query = "SELECT * FROM reminder_audit WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if customer_id:
            query += " AND customer_id = ?"
            params.append(customer_id)
        if reminder_date:
            query += " AND reminder_date = ?"
            params.append(reminder_date)

        query += " ORDER BY updated_at DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_all_audit_records(self) -> pd.DataFrame:
        """Retrieve all raw audit records as a pandas DataFrame."""
        records = self.get_all_reminders()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def get_audit_dataframe(
        self,
        status_filter: Optional[str] = None,
        stage_filter: Optional[str] = None,
        customer_query: Optional[str] = None,
        medicine_query: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve audit history as a formatted pandas DataFrame for UI/export."""
        records = self.get_all_reminders()
        if not records:
            return pd.DataFrame(columns=[
                "Reminder ID", "Customer", "Medicine", "Phone Last-4", "Expected Refill Date",
                "Reminder Stage", "Reminder Date", "Status", "Attempts", "Is Dry Run",
                "Provider Message ID", "Error Details", "Updated At", "Last Attempt At"
            ])

        df = pd.DataFrame(records)

        # Map to user-friendly column names
        display_df = pd.DataFrame()
        display_df["Reminder ID"] = df["reminder_id"]
        display_df["Customer"] = df["customer_name"]
        display_df["Medicine"] = df["item_name"]
        display_df["Phone Last-4"] = df["phone_last4"].apply(lambda p: f"***{p}" if p else "")
        display_df["Expected Refill Date"] = df["expected_refill_date"]
        display_df["Reminder Stage"] = df["reminder_stage"]
        display_df["Reminder Date"] = df["reminder_date"]
        display_df["Status"] = df["status"]
        display_df["Attempts"] = df["attempt_count"]
        display_df["Is Dry Run"] = df["is_dry_run"].apply(lambda d: "Yes" if d else "No")
        display_df["Provider Message ID"] = df["provider_message_id"].fillna("")
        display_df["Error Details"] = df["error_message"].fillna("")
        display_df["Updated At"] = df["updated_at"]
        display_df["Last Attempt At"] = df["last_attempt_at"].fillna("")

        # Keep raw columns for programmatic filtering
        display_df["status_raw"] = df["status"]
        display_df["customer_id"] = df["customer_id"]
        display_df["item_id"] = df["item_id"]

        # Apply in-memory filters
        if status_filter and status_filter != "All":
            display_df = display_df[display_df["Status"].str.lower() == status_filter.lower()]
        if stage_filter and stage_filter != "All":
            display_df = display_df[display_df["Reminder Stage"].astype(str).str.contains(stage_filter.replace("days", "").replace("day", "").strip())]
        if customer_query and customer_query.strip():
            display_df = display_df[display_df["Customer"].str.contains(customer_query.strip(), case=False, na=False)]
        if medicine_query and medicine_query.strip():
            display_df = display_df[display_df["Medicine"].str.contains(medicine_query.strip(), case=False, na=False)]

        return display_df

    def get_dashboard_metrics(self, today_date: Optional[str] = None) -> Dict[str, int]:
        """Compute operational summary counts from persistent state."""
        if today_date is None:
            today_date = date.today().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM reminder_audit WHERE status = 'scheduled'")
            total_scheduled = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM reminder_audit WHERE status = 'accepted' AND is_dry_run = 0")
            total_accepted_live = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM reminder_audit WHERE status = 'accepted' AND is_dry_run = 1")
            total_accepted_dry = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM reminder_audit WHERE status = 'failed'")
            total_failed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM reminder_audit WHERE status = 'cancelled'")
            total_cancelled = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM reminder_audit WHERE status IN ('scheduled', 'sending')")
            total_pending = cursor.fetchone()[0]

            cursor.execute("SELECT COALESCE(SUM(attempt_count), 0) FROM reminder_audit")
            total_attempts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM reminder_audit WHERE reminder_date = ? AND status = 'scheduled'", (today_date,))
            today_due = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM reminder_audit WHERE reminder_date = ? AND status = 'accepted' AND is_dry_run = 0", (today_date,))
            today_accepted = cursor.fetchone()[0]

            return {
                "total_scheduled": int(total_scheduled),
                "total_accepted_live": int(total_accepted_live),
                "total_accepted_dry": int(total_accepted_dry),
                "total_failed": int(total_failed),
                "total_cancelled": int(total_cancelled),
                "total_pending": int(total_pending),
                "total_attempts": int(total_attempts),
                "today_due": int(today_due),
                "today_accepted": int(today_accepted),
            }

    def save_prediction_snapshots(
        self,
        snapshots: Union[pd.DataFrame, List[Dict[str, Any]]],
    ) -> int:
        """Batch save prediction snapshots into persistent storage without overwriting historical records.

        Uses INSERT OR IGNORE on snapshot_id so past snapshot records are strictly preserved.

        Args:
            snapshots: DataFrame or list of prediction dictionaries.

        Returns:
            int: Number of new snapshot records inserted.
        """
        if isinstance(snapshots, pd.DataFrame):
            records = snapshots.to_dict("records")
        elif isinstance(snapshots, (list, tuple)):
            records = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in snapshots]
        else:
            return 0

        if not records:
            return 0

        now_str = _get_utc_now_iso()
        tuples_to_insert = []
        for r in records:
            cid = str(r.get("customer_id", r.get("customerId", ""))).strip()
            iid = str(r.get("item_id", r.get("itemId", ""))).strip()
            if not cid or not iid:
                continue

            cname = str(r.get("customer_name", r.get("customerName", r.get("Customer Name", cid))))
            iname = str(r.get("medication_name", r.get("itemName", r.get("Medication", iid))))
            last_dt = str(r.get("last_purchase_date", r.get("latest_purchase_date", r.get("invoice_date", "")))).strip()
            pred_dt = str(r.get("prediction_date", date.today().isoformat())).strip()
            exp_dt = str(r.get("expected_refill_date", r.get("Expected Refill Date", ""))).strip()
            rem_dt = str(r.get("reminder_date", r.get("Reminder Date", ""))).strip()
            source = str(r.get("prediction_source", "hybrid_supply")).strip()
            status = str(r.get("status", "pending")).strip()

            dos_val = r.get("estimated_days_of_supply", r.get("Estimated Days of Supply", r.get("predicted_days_until_refill")))
            try:
                dos = float(dos_val) if dos_val is not None and not pd.isna(dos_val) else None
            except (ValueError, TypeError):
                dos = None

            batch_val = str(r.get("import_batch_id", "")).strip() or None

            # Deterministic snapshot ID ensuring uniqueness per prediction event
            snap_id = str(r.get("snapshot_id", "")).strip()
            if not snap_id:
                snap_id = f"snap_{cid}_{iid}_{pred_dt}_{last_dt}"

            tuples_to_insert.append((
                snap_id, cid, iid, cname, iname, last_dt, pred_dt, dos, exp_dt, rem_dt, source, status, None, None, now_str, batch_val
            ))

        if not tuples_to_insert:
            return 0

        sql = """
        INSERT OR IGNORE INTO prediction_snapshots (
            snapshot_id, customer_id, item_id, customer_name, item_name,
            last_purchase_date, prediction_date, estimated_days_of_supply,
            expected_refill_date, reminder_date, prediction_source, status,
            actual_next_purchase_date, outcome_error_days, created_at, import_batch_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, tuples_to_insert)
            conn.commit()
            return cursor.rowcount

    def get_prediction_snapshots(
        self,
        customer_id: Optional[str] = None,
        item_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored prediction snapshots with optional filters."""
        query = "SELECT * FROM prediction_snapshots WHERE 1=1"
        params: List[Any] = []

        if customer_id is not None:
            query += " AND customer_id = ?"
            params.append(str(customer_id))
        if item_id is not None:
            query += " AND item_id = ?"
            params.append(str(item_id))
        if status is not None:
            query += " AND status = ?"
            params.append(str(status))

        query += " ORDER BY prediction_date DESC, created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def match_and_update_prediction_outcomes(
        self,
        new_transactions_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Match unresolved prediction snapshots against incoming real-world purchase transactions.

        Compares predicted refill dates against actual next purchase dates and updates
        outcome error metrics without claiming speculative accuracy on current unseen records.

        Args:
            new_transactions_df: DataFrame containing new verified purchase transactions.

        Returns:
            Dict[str, Any]: Summary metrics of evaluated outcomes.
        """
        if new_transactions_df.empty:
            return {"evaluated_count": 0, "pending_count": 0}

        # Normalize incoming columns
        df = new_transactions_df.copy()
        cid_col = "customerId" if "customerId" in df.columns else "customer_id"
        iid_col = "itemId" if "itemId" in df.columns else "item_id"
        dt_col = "invoice_date" if "invoice_date" in df.columns else ("date" if "date" in df.columns else None)

        if not cid_col or not iid_col or not dt_col:
            return {"evaluated_count": 0, "pending_count": 0}

        df[cid_col] = df[cid_col].astype(str)
        df[iid_col] = df[iid_col].astype(str)
        df["_parsed_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
        valid_tx = df[df["_parsed_dt"].notna()]

        # Query all pending snapshots
        pending_snaps = self.get_prediction_snapshots(status="pending")
        if not pending_snaps:
            return {"evaluated_count": 0, "pending_count": 0}

        evaluated_count = 0
        updates_to_run = []

        for snap in pending_snaps:
            s_cid = snap["customer_id"]
            s_iid = snap["item_id"]
            s_last_dt = pd.to_datetime(snap["last_purchase_date"], errors="coerce")
            s_exp_dt = pd.to_datetime(snap["expected_refill_date"], errors="coerce")

            if pd.isna(s_last_dt) or pd.isna(s_exp_dt):
                continue

            # Look for subsequent purchases strictly after last_purchase_date
            matches = valid_tx[
                (valid_tx[cid_col] == s_cid) &
                (valid_tx[iid_col] == s_iid) &
                (valid_tx["_parsed_dt"] > s_last_dt)
            ]

            if not matches.empty:
                earliest_next_dt = matches["_parsed_dt"].min()
                actual_date_str = earliest_next_dt.strftime("%Y-%m-%d")
                error_days = float((earliest_next_dt - s_exp_dt).days)

                updates_to_run.append((
                    actual_date_str,
                    error_days,
                    "evaluated",
                    snap["snapshot_id"],
                ))
                evaluated_count += 1

        if updates_to_run:
            update_sql = """
            UPDATE prediction_snapshots
            SET actual_next_purchase_date = ?,
                outcome_error_days = ?,
                status = ?
            WHERE snapshot_id = ?
            """
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(update_sql, updates_to_run)
                conn.commit()

        remaining_pending = len(pending_snaps) - evaluated_count
        return {
            "evaluated_count": evaluated_count,
            "pending_count": remaining_pending,
        }

    def get_evaluated_prediction_results(self) -> Dict[str, Any]:
        """Retrieve all evaluated prediction snapshots and compute client-facing accuracy metrics.

        Calculates:
        - Predictions evaluated
        - Predictions within ±1 day
        - Predictions within ±3 days
        - Predictions within ±7 days
        - Average difference between expected and actual refill (MAE)

        Returns:
            Dict[str, Any] containing metrics and evaluated DataFrame.
        """
        query = """
        SELECT customer_id, item_id, customer_name, item_name, last_purchase_date,
               expected_refill_date, reminder_date, actual_next_purchase_date, outcome_error_days
        FROM prediction_snapshots
        WHERE status = 'evaluated' AND actual_next_purchase_date IS NOT NULL
        ORDER BY last_purchase_date DESC, expected_refill_date DESC
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            raw_evals = [dict(r) for r in rows]

        if not raw_evals:
            return {
                "status": "no_records",
                "predictions_evaluated": 0,
                "within_1_day_count": 0,
                "within_1_day_pct": 0.0,
                "within_3_days_count": 0,
                "within_3_days_pct": 0.0,
                "within_7_days_count": 0,
                "within_7_days_pct": 0.0,
                "mean_absolute_error": 0.0,
                "evaluated_df": pd.DataFrame(columns=[
                    "Customer Name", "Medication", "Last Purchase Date",
                    "Expected Refill Date", "Actual Purchase Date",
                    "Difference (Days)", "Within ±3 Days", "Within ±7 Days"
                ]),
            }

        df_eval = pd.DataFrame(raw_evals)
        df_eval["abs_error"] = df_eval["outcome_error_days"].abs()

        n_eval = len(df_eval)
        w1_cnt = int((df_eval["abs_error"] <= 1.0).sum())
        w3_cnt = int((df_eval["abs_error"] <= 3.0).sum())
        w7_cnt = int((df_eval["abs_error"] <= 7.0).sum())
        mae = float(df_eval["abs_error"].mean())

        # Build clean client-facing table
        client_table = pd.DataFrame()
        client_table["Customer Name"] = df_eval["customer_name"].fillna(df_eval["customer_id"])
        client_table["Medication"] = df_eval["item_name"].fillna(df_eval["item_id"])
        client_table["Last Purchase Date"] = df_eval["last_purchase_date"]
        client_table["Expected Refill Date"] = df_eval["expected_refill_date"]
        client_table["Actual Purchase Date"] = df_eval["actual_next_purchase_date"]

        def _fmt_diff(d: float) -> str:
            if pd.isna(d):
                return "-"
            val = int(round(d))
            return f"+{val} days" if val > 0 else (f"{val} days" if val < 0 else "0 days (exact)")

        client_table["Difference (Days)"] = df_eval["outcome_error_days"].apply(_fmt_diff)
        client_table["Within ±3 Days"] = df_eval["abs_error"].apply(lambda x: "Yes" if x <= 3.0 else "No")
        client_table["Within ±7 Days"] = df_eval["abs_error"].apply(lambda x: "Yes" if x <= 7.0 else "No")

        return {
            "status": "success",
            "predictions_evaluated": n_eval,
            "within_1_day_count": w1_cnt,
            "within_1_day_pct": round(w1_cnt / n_eval * 100, 1),
            "within_3_days_count": w3_cnt,
            "within_3_days_pct": round(w3_cnt / n_eval * 100, 1),
            "within_7_days_count": w7_cnt,
            "within_7_days_pct": round(w7_cnt / n_eval * 100, 1),
            "mean_absolute_error": round(mae, 1),
            "evaluated_df": client_table,
        }

    def save_import_batch(self, batch_info: Dict[str, Any]) -> None:
        """Save import batch record to persistent storage."""
        sql = """
        INSERT OR REPLACE INTO import_batches (
            import_batch_id, upload_timestamp, source_filename, detected_date_range,
            record_count, records_inserted, records_skipped, records_requiring_review,
            processing_status, created_at, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (
                str(batch_info.get("import_batch_id", "")),
                str(batch_info.get("upload_timestamp", _get_utc_now_iso())),
                str(batch_info.get("source_filename", "unknown")),
                str(batch_info.get("detected_date_range", "-")),
                int(batch_info.get("record_count", 0)),
                int(batch_info.get("records_inserted", 0)),
                int(batch_info.get("records_skipped", 0)),
                int(batch_info.get("records_requiring_review", 0)),
                str(batch_info.get("processing_status", "completed")),
                str(batch_info.get("created_at", _get_utc_now_iso())),
                int(batch_info.get("is_active", 1)),
            ))
            conn.commit()

    def get_import_batches(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieve recorded import batches."""
        query = "SELECT * FROM import_batches WHERE 1=1"
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY created_at DESC"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(r) for r in cursor.fetchall()]

    def get_latest_active_import_batch(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent active import batch."""
        batches = self.get_import_batches(active_only=True)
        return batches[0] if batches else None

    def rollback_import_batch(self, import_batch_id: str) -> Dict[str, Any]:
        """Rollback an import batch: deactivate batch and invalidate associated prediction snapshots."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE import_batches SET is_active = 0, processing_status = 'rolled_back' WHERE import_batch_id = ?", (import_batch_id,))
            # Delete or invalidate snapshots generated under this batch
            cursor.execute("DELETE FROM prediction_snapshots WHERE import_batch_id = ?", (import_batch_id,))
            snaps_removed = cursor.rowcount
            conn.commit()
            return {
                "import_batch_id": import_batch_id,
                "status": "rolled_back",
                "snapshots_invalidated": snaps_removed,
            }

