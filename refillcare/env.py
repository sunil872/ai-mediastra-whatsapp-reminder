"""Load the project ``.env`` once, before any ``REFILLCARE_*`` / ``ML_*`` switch is read.

Deployment environment variables win over ``.env`` (same policy as the WhatsApp client).
"""
from __future__ import annotations

from pathlib import Path

_LOADED = False
DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def ensure_dotenv() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=DOTENV_PATH, override=False)
    except ImportError:
        pass
