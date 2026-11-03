"""Control-plane tables shared by every store: the store registry and the background job queue."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import ControlBase


class StoreModel(ControlBase):
    """A pharmacy store served by this deployment (its data lives in its own schema / database file)."""
    __tablename__ = "refillcare_stores"

    store_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    store_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    store_contact: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Per-store switches (patient_identity, item_grouping, rules, ...) override the .env defaults
    settings: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)  # active | disabled
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class JobModel(ControlBase):
    """Background work (retrain, rule A/B, rolling test, re-key, post-upload sync retry).

    Jobs survive restarts and run once: a worker claims a queued job atomically, so several API
    workers / containers never run the same job twice.
    """
    __tablename__ = "refillcare_jobs"
    __table_args__ = (Index("idx_jobs_status_created", "status", "created_at"),)

    job_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued", nullable=False)  # queued | running | done | failed
    dedupe_key: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64), default="system", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
