"""Module for fetching pharmacy sales history with caching and multi-store readiness."""

from __future__ import annotations

from typing import Optional
from datetime import date, datetime
import pandas as pd
from sqlalchemy.orm import Session

from config.config import PURCHASE_HISTORY_PATH, CLEAN_TRANSACTIONS_PATH
from database.models import SalesTransactionModel


def fetch_purchase_history(
    db: Optional[Session] = None,
    store_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch longitudinal purchase history from parquet store or SQL database.

    Args:
        db: Optional database session.
        store_id: Optional store/pharmacy ID for multi-store filtering.
        start_date: Optional minimum invoice date.
        end_date: Optional maximum invoice date.

    Returns:
        pd.DataFrame: Clean purchase history dataframe.
    """
    # 1. Attempt reading from fast parquet store
    if PURCHASE_HISTORY_PATH.exists():
        df = pd.read_parquet(PURCHASE_HISTORY_PATH)
    elif CLEAN_TRANSACTIONS_PATH.exists():
        df = pd.read_parquet(CLEAN_TRANSACTIONS_PATH)
    elif db is not None and db.bind is not None:
        from sqlalchemy import text
        df = pd.read_sql(text("SELECT * FROM sales_transactions"), db.bind)
    else:
        df = pd.DataFrame()

    if df.empty:
        return df

    # Defense-in-depth: Exclude B2B / Inter-Store (SB/) transactions from customer purchase history
    if "invoice_number" in df.columns or "refillcare_eligible" in df.columns:
        from refillcare.data.transaction_classifier import filter_eligible_customer_transactions
        df = filter_eligible_customer_transactions(df, invoice_col="invoice_number")

    # Apply date filters if requested
    if "invoice_date" in df.columns:
        df["_dt"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        if start_date:
            df = df.loc[df["_dt"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df.loc[df["_dt"] <= pd.Timestamp(end_date)]
        df = pd.DataFrame(df).drop(columns=["_dt"], errors="ignore")

    return pd.DataFrame(df)
