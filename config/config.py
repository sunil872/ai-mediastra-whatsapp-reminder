"""Centralized Configuration Module for Medical Reminder AI / RefillCare."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# Application & Environment
APP_NAME = "Medical Reminder AI / RefillCare"
APP_VERSION = "2.0.0"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1")

# Standard Date Formatting
UI_DATE_FORMAT = "%d-%m-%Y"
DB_DATE_FORMAT = "%Y-%m-%d"
SUPPORTED_SOURCE_DATE_FORMATS = [
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
]

# Database Configuration (PostgreSQL with SQLite fallback)
DATABASE_URL = os.getenv("DATABASE_URL")
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "refillcare" / "processed" / "enterprise.db"
DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"

# Storage & Parquet Artifact Paths
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "refillcare" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

PURCHASE_HISTORY_PATH = PROCESSED_DATA_DIR / "purchase_history.parquet"
CLEAN_TRANSACTIONS_PATH = PROCESSED_DATA_DIR / "clean_transactions.parquet"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.parquet"

# Models Directory (.pkl model bundles)
MODELS_DIR = PROCESSED_DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PKL_MODEL_PATH = MODELS_DIR / "refill_model.pkl"
DEFAULT_JOBLIB_MODEL_PATH = MODELS_DIR / "refill_model.joblib"

# Reminder & Refill Strategy Settings
PRIMARY_REMINDER_BUFFER_DAYS = 2
REMINDER_STAGES_DAYS = [-7, -3, -1, 0, 2, 5, 40]  # 7-Stage Refill Lifecycle
CHURN_CUTOFF_DAYS = 75  # Hard churn cutoff post-due date
MIN_PURCHASES_FOR_PATH_A = 6  # Established recurring history threshold

# WhatsApp Gateway (Xinno)
XINNO_API_URL = os.getenv("XINNO_API_URL", "https://api.xinno.in/v1/messages")
XINNO_API_KEY = os.getenv("XINNO_API_KEY", "")
WHATSAPP_TEMPLATE_NAME = os.getenv("WHATSAPP_TEMPLATE_NAME", "refill_reminder_v1")
WHATSAPP_DRY_RUN = os.getenv("WHATSAPP_DRY_RUN", "True").lower() in ("true", "1")  # Strictly True by default

# Logging
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
WHATSAPP_LOG_PATH = LOGS_DIR / "whatsapp_send.log"
