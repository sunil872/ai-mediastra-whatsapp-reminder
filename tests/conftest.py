"""
Pytest fixtures and path setup for the AI Mediastra test suite.

Project root is added to sys.path so `utils` and `services` imports resolve
when tests live under tests/.

Isolation: before anything imports ``database.connection``, the database is pointed at a temporary
SQLite file and the environment is set to ``test``, so no test can read or write the live
``enterprise.db`` (store registry, job queue and store tables all live in the temp file).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="refillcare_test_db_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_DB_DIR / 'enterprise_test.db').as_posix()}"
os.environ["REFILLCARE_ENV"] = "test"
os.environ.setdefault("REFILLCARE_STORE_ID", "store-1")
# Fixtures use past sales dates; the stale-data send guard has its own tests that switch it on
os.environ["REFILLCARE_STALE_DATA_GUARD"] = "false"
for _k in ("REFILLCARE_API_KEY", "WHATSAPP_WEBHOOK_SECRET", "REFILLCARE_UI_PASSWORD", "REFILLCARE_MODEL_BUCKET"):
    os.environ.pop(_k, None)
