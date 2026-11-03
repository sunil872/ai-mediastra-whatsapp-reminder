"""Logging for every process (API, worker, Streamlit): one handler, JSON lines, phones masked.

- JSON lines on stdout (``REFILLCARE_LOG_FORMAT=json``, the default in production) so CloudWatch /
  any collector can index them; plain text otherwise.
- Every message passes a PHI filter that masks Indian mobile numbers (``98480*****``-style:
  first 2 + last 2 digits kept) - a log line never carries a full phone number.
- ``request_id`` (set by the API middleware) is attached to every record.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str] = ContextVar("refillcare_request_id", default="-")
# An Indian mobile written any common way: 9848012345, +91 98480 12345, 98480-12345, 0 9848012345
_PHONE_RE = re.compile(r"(?<![\d+])(?:\+?91[\s-]?|0)?[6-9](?:[\s-]?\d){9}(?!\d)")
_configured = False


def mask_phones(text: str) -> str:
    """'9848012345' -> '98******45' (also '+91 98480 12345')."""
    def _mask(m: "re.Match[str]") -> str:
        d = re.sub(r"\D", "", m.group(0))[-10:]
        return f"{d[:2]}******{d[-2:]}"

    return _PHONE_RE.sub(_mask, text)


def mask_phone(phone) -> str:
    digits = re.sub(r"\D", "", str(phone or ""))
    return f"{digits[-10:-8]}******{digits[-2:]}" if len(digits) >= 10 else ("*" * len(digits))


class PHIFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        masked = mask_phones(msg)
        if masked != msg:
            record.msg, record.args = masked, ()
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        try:
            from refillcare.tenancy import current_store_id

            out["store_id"] = current_store_id()
        except Exception:
            pass
        if record.exc_info:
            out["exc"] = mask_phones(self.formatException(record.exc_info))
        return json.dumps(out, ensure_ascii=False)


def configure_logging(force: bool = False) -> None:
    global _configured
    if _configured and not force:
        return
    from refillcare.tenancy import is_production

    fmt = os.getenv("REFILLCARE_LOG_FORMAT", "json" if is_production() else "text").lower()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(PHIFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else
                         logging.Formatter("%(asctime)s %(levelname)s [%(name)s] [%(request_id)s] %(message)s"))
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(os.getenv("REFILLCARE_LOG_LEVEL", "INFO").upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _configured = True
