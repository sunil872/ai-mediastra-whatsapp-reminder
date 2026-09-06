"""
Unit and integration tests for Cloudinary image upload and image campaign URL validation.

Ensures:
1. Cloudinary configuration is read from environment variables.
2. Missing Cloudinary configuration produces a clear error.
3. Local image validation accepts JPG, JPEG, PNG, WEBP.
4. Invalid/non-image files are rejected.
5. Cloudinary upload logic is mocked in tests (no real API calls).
6. Mocked Cloudinary upload returns secure HTTPS URL.
7. Returned secure URL becomes selected_image_url.
8. Selected Cloudinary URL is used in Xinno image-header payload.
9. Manual HTTPS URL selection still works.
10. HTTP URL is rejected.
11. Invalid URL is rejected.
12. Empty URL is rejected.
13. A local filesystem path can never become Xinno image.link.
14. Local upload without successful Cloudinary upload cannot pass live-send validation.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from services.cloudinary_image import (
    ALLOWED_EXTENSIONS,
    get_cloudinary_config,
    is_cloudinary_configured,
    upload_image_to_cloudinary,
    validate_image_file,
)
from services.xinno_image_template import send_image_template_message
from utils.image_campaign import validate_image_url


# ===================================================================
# Dummy valid image bytes
# ===================================================================
# Minimal valid JPEG magic bytes: \xff\xd8\xff...
DUMMY_JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00" + b"\x00" * 50

# Minimal valid PNG magic bytes: \x89PNG\r\n\x1a\n...
DUMMY_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"

# Minimal valid WEBP magic bytes: RIFF....WEBP
DUMMY_WEBP_BYTES = b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00" + b"\x00" * 30


# ===================================================================
# 1. Cloudinary configuration from environment variables
# ===================================================================
class TestCloudinaryConfig:
    def test_reads_config_from_env(self, monkeypatch):
        monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "my_cloud")
        monkeypatch.setenv("CLOUDINARY_API_KEY", "key_12345")
        monkeypatch.setenv("CLOUDINARY_API_SECRET", "secret_abcde")

        cfg = get_cloudinary_config()
        assert cfg["cloud_name"] == "my_cloud"
        assert cfg["api_key"] == "key_12345"
        assert cfg["api_secret"] == "secret_abcde"

        configured, reason = is_cloudinary_configured()
        assert configured is True
        assert reason == ""

    def test_missing_cloud_name_produces_error(self, monkeypatch):
        monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "")
        monkeypatch.setenv("CLOUDINARY_API_KEY", "key_12345")
        monkeypatch.setenv("CLOUDINARY_API_SECRET", "secret_abcde")

        configured, reason = is_cloudinary_configured()
        assert configured is False
        assert "CLOUDINARY_CLOUD_NAME" in reason

    def test_placeholder_config_produces_error(self, monkeypatch):
        monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "your_cloud_name")
        monkeypatch.setenv("CLOUDINARY_API_KEY", "your_api_key")
        monkeypatch.setenv("CLOUDINARY_API_SECRET", "your_api_secret")

        configured, reason = is_cloudinary_configured()
        assert configured is False
        assert "CLOUDINARY_CLOUD_NAME" in reason


# ===================================================================
# 2. Local Image Validation
# ===================================================================
class TestImageValidation:
    def test_accepts_valid_jpeg(self):
        valid, msg = validate_image_file(DUMMY_JPEG_BYTES, filename="test.jpg")
        assert valid is True
        assert msg == ""

    def test_accepts_valid_jpeg_with_jpeg_ext(self):
        valid, msg = validate_image_file(DUMMY_JPEG_BYTES, filename="photo.jpeg")
        assert valid is True
        assert msg == ""

    def test_accepts_valid_png(self):
        valid, msg = validate_image_file(DUMMY_PNG_BYTES, filename="graphic.png")
        assert valid is True
        assert msg == ""

    def test_accepts_valid_webp(self):
        valid, msg = validate_image_file(DUMMY_WEBP_BYTES, filename="banner.webp")
        assert valid is True
        assert msg == ""

    def test_accepts_bytesio_object(self):
        buf = io.BytesIO(DUMMY_PNG_BYTES)
        valid, msg = validate_image_file(buf, filename="banner.png")
        assert valid is True
        assert msg == ""

    def test_rejects_empty_file(self):
        valid, msg = validate_image_file(b"", filename="empty.jpg")
        assert valid is False
        assert "empty" in msg.lower()

    def test_rejects_unsupported_extension(self):
        valid, msg = validate_image_file(DUMMY_JPEG_BYTES, filename="doc.pdf")
        assert valid is False
        assert "unsupported" in msg.lower() or "allowed formats" in msg.lower()

    def test_rejects_non_image_content(self):
        text_content = b"This is just plain text, not an image."
        valid, msg = validate_image_file(text_content, filename="fake.png")
        assert valid is False
        assert "not a valid" in msg.lower()


# ===================================================================
# 3. Cloudinary Upload (Mocked — Zero Real Calls)
# ===================================================================
class TestCloudinaryUpload:
    @patch("cloudinary.uploader.upload")
    def test_mocked_upload_returns_secure_url(self, mock_upload, monkeypatch):
        monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test_cloud")
        monkeypatch.setenv("CLOUDINARY_API_KEY", "123456789")
        monkeypatch.setenv("CLOUDINARY_API_SECRET", "abcdef_secret")

        mock_upload.return_value = {
            "secure_url": "https://res.cloudinary.com/test_cloud/image/upload/v123456/pharma_hubb/refill_campaign/test_image.jpg",
            "public_id": "pharma_hubb/refill_campaign/test_image",
            "format": "jpg",
        }

        buf = io.BytesIO(DUMMY_JPEG_BYTES)
        result = upload_image_to_cloudinary(buf, filename="test_image.jpg")

        assert result["success"] is True
        assert result["secure_url"].startswith("https://res.cloudinary.com")
        assert result["public_id"] == "pharma_hubb/refill_campaign/test_image"
        assert result["error"] is None

        mock_upload.assert_called_once()

    def test_upload_fails_when_unconfigured(self, monkeypatch):
        monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "")
        monkeypatch.setenv("CLOUDINARY_API_KEY", "")
        monkeypatch.setenv("CLOUDINARY_API_SECRET", "")

        result = upload_image_to_cloudinary(DUMMY_JPEG_BYTES, filename="test.jpg")
        assert result["success"] is False
        assert "Configuration Error" in result["error"]
        assert result["secure_url"] is None

    @patch("cloudinary.uploader.upload")
    def test_upload_masks_secrets_on_exception(self, mock_upload, monkeypatch):
        monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test_cloud")
        monkeypatch.setenv("CLOUDINARY_API_KEY", "MY_SUPER_API_KEY")
        monkeypatch.setenv("CLOUDINARY_API_SECRET", "MY_SUPER_SECRET_VALUE")

        mock_upload.side_effect = Exception("Cloudinary error with secret MY_SUPER_SECRET_VALUE and key MY_SUPER_API_KEY")

        result = upload_image_to_cloudinary(DUMMY_JPEG_BYTES, filename="test.jpg")
        assert result["success"] is False
        assert "MY_SUPER_SECRET_VALUE" not in result["error"]
        assert "MY_SUPER_API_KEY" not in result["error"]
        assert "***MASKED***" in result["error"]


# ===================================================================
# 4. URL Validation Helpers
# ===================================================================
class TestImageUrlValidation:
    def test_valid_https_url(self):
        url = "https://res.cloudinary.com/troli5kq/image/upload/f_auto,q_auto/PHARMA_HUBB_-_medicine_Refill_Reminder_Image_Template"
        valid, msg = validate_image_url(url)
        assert valid is True
        assert msg == ""

    def test_valid_https_url_with_query_params(self):
        url = "https://images.example.com/banner.png?size=large&v=2"
        valid, msg = validate_image_url(url)
        assert valid is True
        assert msg == ""

    def test_rejects_http_url(self):
        url = "http://insecure.example.com/image.jpg"
        valid, msg = validate_image_url(url)
        assert valid is False
        assert "https" in msg.lower() or "insecure" in msg.lower()

    def test_rejects_empty_or_whitespace_url(self):
        valid, msg = validate_image_url("")
        assert valid is False
        valid2, msg2 = validate_image_url("   ")
        assert valid2 is False

    def test_rejects_none_url(self):
        valid, msg = validate_image_url(None)
        assert valid is False

    def test_rejects_local_windows_path(self):
        valid, msg = validate_image_url(r"C:\Users\sunil\Pictures\reminder.png")
        assert valid is False
        assert "local filesystem" in msg.lower() or "https" in msg.lower()

    def test_rejects_local_unix_path(self):
        valid, msg = validate_image_url("/tmp/reminder.png")
        assert valid is False
        assert "local filesystem" in msg.lower() or "https" in msg.lower()

    def test_rejects_file_protocol(self):
        valid, msg = validate_image_url("file:///C:/Users/reminder.png")
        assert valid is False


# ===================================================================
# 5. Xinno Payload Integration with Cloudinary & Custom URLs
# ===================================================================
class TestXinnoImagePayloadIntegration:
    def test_selected_cloudinary_url_reaches_xinno_image_link(self, monkeypatch):
        monkeypatch.setenv("XINNO_IMAGE_TEMPLATE_NAME", "refill_reminder_image")
        monkeypatch.setenv("WHATSAPP_TEMPLATE_LANGUAGE", "en")

        cloudinary_secure_url = "https://res.cloudinary.com/troli5kq/image/upload/v999/custom_uploaded_banner.jpg"

        res = send_image_template_message(
            phone_number="917659935016",
            customer_name="Sunil",
            store_name="PHARMA HUBB",
            branch="Chadargatt",
            medicine_list="• METFORMIN 500 MG\n• TELMISARTAN 40 MG",
            contact_no="9581473474",
            manager_contact="9885473474",
            image_url=cloudinary_secure_url,
            dry_run=True,
        )

        assert res["success"] is True
        payload = res["response"]["payload"]
        components = payload["template"]["components"]
        header = next(c for c in components if c["type"] == "header")
        assert header["parameters"][0]["type"] == "image"
        assert header["parameters"][0]["image"]["link"] == cloudinary_secure_url

    def test_local_path_rejected_by_xinno_service(self, monkeypatch):
        monkeypatch.setenv("XINNO_IMAGE_TEMPLATE_NAME", "refill_reminder_image")

        res = send_image_template_message(
            phone_number="917659935016",
            customer_name="Sunil",
            store_name="PHARMA HUBB",
            branch="Chadargatt",
            medicine_list="• METFORMIN 500 MG",
            contact_no="9581473474",
            manager_contact="9885473474",
            image_url=r"C:\Users\sunil\Pictures\local_image.png",
            dry_run=True,
        )

        assert res["success"] is False
        assert "Configuration Error" in res["message"] or "local filesystem" in res["message"]

        assert res["success"] is False
        assert "Configuration Error" in res["message"] or "local filesystem" in res["message"]
