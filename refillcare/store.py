"""Single-store configuration for RefillCare v1 (NGOs Colony).

Everything store-specific comes from ``.env`` so nothing is hard-coded:
- ``MEDICAL_STORE_NAME`` / ``MEDICAL_STORE_CONTACT``: used in WhatsApp messages.
- ``REFILLCARE_DATA_DIR``: data folder (default ``data/refillcare``) holding the master sales CSV,
  ``processed/`` parquets, models and backups.

v2 (Mediastra integration) will resolve tenant / store / org from the Mediastra login instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config.config import PROJECT_ROOT


@dataclass(frozen=True)
class StoreConfig:
    data_dir: Path
    store_name: str
    store_contact: str
    store_id: str = "store-1"

    @classmethod
    def from_env(cls) -> "StoreConfig":
        data_dir = Path(os.getenv("REFILLCARE_DATA_DIR", "") or (PROJECT_ROOT / "data" / "refillcare"))
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        return cls(
            data_dir=data_dir,
            store_name=(os.getenv("MEDICAL_STORE_NAME", "") or "Pharmacy").strip(),
            store_contact=os.getenv("MEDICAL_STORE_CONTACT", "").strip(),
            store_id=(os.getenv("REFILLCARE_STORE_ID", "") or "store-1").strip(),
        )

    @property
    def raw_backup_dir(self) -> Path:
        return self.data_dir / "raw_backup"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def history_parquet(self) -> Path:
        return self.processed_dir / "purchase_history.parquet"

    @property
    def clean_transactions_parquet(self) -> Path:
        return self.processed_dir / "clean_transactions.parquet"

    @property
    def master_csv(self) -> Path:
        return self.data_dir / "customer_data_fields.csv"

    @property
    def legacy_sqlite(self) -> Path:
        return self.processed_dir / "refillcare.db"

    def ensure(self) -> "StoreConfig":
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        return self
