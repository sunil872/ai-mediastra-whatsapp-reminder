"""
Cloudinary Image Upload Service

Handles safe configuration, validation, and uploading of campaign images
to Cloudinary, returning the secure HTTPS URL for use in WhatsApp template headers.

NEVER prints, logs, or exposes CLOUDINARY_API_SECRET.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from dotenv import load_dotenv

logger = logging.getLogger("cloudinary_image")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)

# ---------------------------------------------------------------------------
# Allowed image extensions & MIME signatures
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_CLOUDINARY_FOLDER = "pharma_hubb/refill_campaign"
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


def get_cloudinary_config(reload_dotenv: bool = False) -> Dict[str, str]:
    """
    Read Cloudinary configuration from environment variables.

    Returns:
        Dict with keys: cloud_name, api_key, api_secret.
    """
    if reload_dotenv:
        load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
    api_key = os.getenv("CLOUDINARY_API_KEY", "").strip()
    api_secret = os.getenv("CLOUDINARY_API_SECRET", "").strip()

    return {
        "cloud_name": cloud_name,
        "api_key": api_key,
        "api_secret": api_secret,
    }


def is_cloudinary_configured(reload_dotenv: bool = False) -> Tuple[bool, str]:
    """
    Check if Cloudinary environment variables are set and valid.

    Returns:
        (is_configured, reason_if_not)
    """
    cfg = get_cloudinary_config(reload_dotenv=reload_dotenv)
    missing = []
    if not cfg["cloud_name"] or cfg["cloud_name"].startswith("your_"):
        missing.append("CLOUDINARY_CLOUD_NAME")
    if not cfg["api_key"] or cfg["api_key"].startswith("your_"):
        missing.append("CLOUDINARY_API_KEY")
    if not cfg["api_secret"] or cfg["api_secret"].startswith("your_"):
        missing.append("CLOUDINARY_API_SECRET")

    if missing:
        return False, f"Missing or placeholder Cloudinary configuration: {', '.join(missing)}"
    return True, ""


def validate_image_file(
    file_data: Union[bytes, io.BytesIO, Any],
    filename: str = "",
) -> Tuple[bool, str]:
    """
    Validate that an uploaded file is a supported, non-empty image (JPG, JPEG, PNG, WEBP).

    Args:
        file_data: Raw bytes, BytesIO, or Streamlit UploadedFile.
        filename: Original filename (used for extension check).

    Returns:
        (is_valid, error_message)
    """
    # 1. Extension check if filename provided
    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file format '{ext}'. Allowed formats: JPG, JPEG, PNG, WEBP."

    # 2. Extract bytes
    raw_bytes: bytes = b""
    if isinstance(file_data, bytes):
        raw_bytes = file_data
    elif hasattr(file_data, "getvalue"):
        raw_bytes = file_data.getvalue()
    elif hasattr(file_data, "read"):
        raw_bytes = file_data.read()
        if hasattr(file_data, "seek"):
            file_data.seek(0)
    else:
        return False, "Invalid file object provided."

    # 3. Size check
    if not raw_bytes or len(raw_bytes) == 0:
        return False, "The selected file is empty."

    if len(raw_bytes) > MAX_IMAGE_SIZE_BYTES:
        return False, f"Image size exceeds the maximum allowed limit of {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB."

    # 4. Content signature (magic bytes) validation
    # JPEG: starts with \xff\xd8\xff
    # PNG: starts with \x89PNG\r\n\x1a\n
    # WEBP: starts with RIFF....WEBP
    is_jpeg = raw_bytes.startswith(b"\xff\xd8\xff")
    is_png = raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP"

    if not (is_jpeg or is_png or is_webp):
        return False, "File content is not a valid JPG, PNG, or WEBP image."

    return True, ""


def _mask_secret(text: str, secret: str) -> str:
    """Safely mask secret from any error message or string."""
    if not secret or not text:
        return text
    return text.replace(secret, "***MASKED***")


def upload_image_to_cloudinary(
    file_obj: Union[bytes, io.BytesIO, Any],
    filename: str = "",
    folder: str = DEFAULT_CLOUDINARY_FOLDER,
    reload_dotenv: bool = False,
) -> Dict[str, Any]:
    """
    Upload a local image to Cloudinary and return the secure HTTPS URL.

    Args:
        file_obj: File bytes, file-like object, or Streamlit UploadedFile.
        filename: Original file name.
        folder: Cloudinary destination folder (default: 'pharma_hubb/refill_campaign').
        reload_dotenv: If True, reload .env before checking config.

    Returns:
        Dict:
            {
                "success": bool,
                "secure_url": Optional[str],
                "public_id": Optional[str],
                "format": Optional[str],
                "error": Optional[str],
            }
    """
    # 1. Validate file content
    valid, val_err = validate_image_file(file_obj, filename=filename)
    if not valid:
        return {
            "success": False,
            "secure_url": None,
            "public_id": None,
            "format": None,
            "error": f"Image Validation Failed: {val_err}",
        }

    # 2. Check configuration
    is_configured, cfg_err = is_cloudinary_configured(reload_dotenv=reload_dotenv)
    if not is_configured:
        return {
            "success": False,
            "secure_url": None,
            "public_id": None,
            "format": None,
            "error": f"Cloudinary Configuration Error: {cfg_err}. Please update your .env file.",
        }

    cfg = get_cloudinary_config(reload_dotenv=reload_dotenv)

    # 3. Configure and perform upload via official Cloudinary SDK
    try:
        import cloudinary
        import cloudinary.uploader

        cloudinary.config(
            cloud_name=cfg["cloud_name"],
            api_key=cfg["api_key"],
            api_secret=cfg["api_secret"],
            secure=True,
        )

        # Ensure file pointer is at start if seekable
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        # Upload
        upload_result = cloudinary.uploader.upload(
            file_obj,
            folder=folder,
            resource_type="image",
            use_filename=True,
            unique_filename=True,
            overwrite=False,
        )

        secure_url = upload_result.get("secure_url") or upload_result.get("url")
        public_id = upload_result.get("public_id")
        img_format = upload_result.get("format")

        if not secure_url or not str(secure_url).startswith("https://"):
            return {
                "success": False,
                "secure_url": None,
                "public_id": public_id,
                "format": img_format,
                "error": "Cloudinary upload did not return a valid HTTPS secure_url.",
            }

        return {
            "success": True,
            "secure_url": secure_url,
            "public_id": public_id,
            "format": img_format,
            "error": None,
        }

    except Exception as exc:
        raw_msg = str(exc)
        safe_msg = _mask_secret(raw_msg, cfg.get("api_secret", ""))
        safe_msg = _mask_secret(safe_msg, cfg.get("api_key", ""))
        logger.error(f"Cloudinary upload failed: {safe_msg}")
        return {
            "success": False,
            "secure_url": None,
            "public_id": None,
            "format": None,
            "error": f"Cloudinary Upload Error: {safe_msg}",
        }
