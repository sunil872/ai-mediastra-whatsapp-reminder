"""
Pytest fixtures and path setup for the AI Mediastra test suite.

Project root is added to sys.path so `utils` and `services` imports resolve
when tests live under tests/.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
