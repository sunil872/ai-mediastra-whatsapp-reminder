"""Legacy Entry Point & Reverse Proxy Wrapper.

This module exposes the unified FastAPI application instance from `api.main`,
ensuring full backward compatibility for deployments targeting `app.main:app`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app

__all__ = ["app"]
