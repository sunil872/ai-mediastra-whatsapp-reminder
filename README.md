# PHARMA HUBB WhatsApp Campaign Center

A Streamlit application for bulk WhatsApp medicine refill reminders. Operators upload customer lists, validate recipients, send an approved Xinno template, and track per-customer results.

Built for **AI Mediastra / PHARMA HUBB**.

---

## Overview

Pharmacy teams need a controlled way to message refill follow-ups without sending from spreadsheets manually. This app provides a single bulk workflow:

1. Upload a CSV or Excel customer list  
2. Validate names and Indian mobile numbers  
3. Preview eligible recipients and template variables  
4. Confirm, then send sequentially through Xinno  
5. Review results and export an audit CSV  

Upload and validation never send messages. Live sends require an explicit confirmation checkbox.

---

## Key features

- CSV and XLSX customer uploads  
- Flexible column headers (e.g. `customer_name` / `mobile` map to `Name` / `Phone number`)  
- Name and phone validation  
- Indian mobile normalization to `91XXXXXXXXXX`  
- Duplicate detection after normalization  
- Eligible-recipient filtering (`Status = Valid` only)  
- Dynamic template preview per customer  
- Xinno WhatsApp CPaaS integration  
- Bulk sequential sending (no parallel fan-out, no auto-retry)  
- Bulk attempt IDs with Streamlit re-run / double-click protection  
- Per-customer audit records (masked phones, no API keys)  
- Session send history and CSV result export  
- Environment-based configuration (`.env`)  
- Automated pytest suite (mocked Xinno; no live sends in tests)

---

## Workflow

```text
Upload Customer List (CSV / XLSX)
        ↓
Normalize Column Headers
        ↓
Validate Data
        ↓
Normalize Phone Numbers
        ↓
Identify Eligible Customers
        ↓
Preview Template & Samples
        ↓
Confirm Campaign
        ↓
Bulk WhatsApp Send (sequential)
        ↓
Track Results
        ↓
Export Report / View Session History
```

---

## How the system works

### Customer data

Files must provide a name column and a phone column. Canonical headers are `Name` and `Phone number`. Common aliases are accepted automatically, for example:

| Role | Accepted headers (examples) |
|------|-----------------------------|
| Name | `Name`, `customer_name`, `full_name`, `fullname` |
| Phone | `Phone number`, `phone`, `mobile`, `whatsapp number` |

Extra columns are ignored for messaging.

### Validation

Rows are classified as:

- **Valid** — usable name + valid Indian mobile  
- **Invalid** — missing name/phone or bad number  
- **Duplicate** — same normalized phone as an earlier valid row  

Only **Valid** customers are eligible to send.

### Phone normalization

Indian mobiles are cleaned and normalized before Xinno:

| Input | Normalized |
|-------|------------|
| `7659935016` | `917659935016` |
| `+91 76599 35016` | `917659935016` |
| `9176599 35016` | `917659935016` |

Duplicate checks run **after** normalization.

### WhatsApp messaging

Messages are sent through the existing Xinno REST client (`services/xinno_whatsapp.py`) using the approved template. Credentials come from environment variables. API keys are masked in UI/logs and never written to audit exports.

Default send mode for the service is `dry_run=True`. Live production bulk send uses `dry_run=False` only after the operator confirms in the UI.

### Bulk campaigns

Eligible customers are processed one at a time. Each customer gets an independent payload (name + phone from the same row). Failures are recorded and the batch continues. The same bulk attempt ID cannot execute twice in a Streamlit session.

**Note:** Streamlit “Accepted” means Xinno/Meta accepted the API request (`wamid`). Phone delivery is confirmed in the Xinno/Meta dashboard or webhooks.

---

## WhatsApp template

| Setting | Value |
|---------|--------|
| Template | `reminder_refill_followup_v3` |
| Language | `en` (policy: deterministic) |
| `{{1}}` | Customer name |
| `{{2}}` | Medical store name (default `PHARMA HUBB`) |
| `{{3}}` | Medical store name (default `PHARMA HUBB`) |

---

## Technology stack

- Python 3.10+  
- Streamlit  
- Pandas  
- Requests  
- python-dotenv  
- openpyxl (XLSX)  
- pytest  
- Xinno WhatsApp CPaaS  

---

## Project structure

```text
ai-mediastra-whatsapp-reminder/
├── app.py                 # Streamlit UI (bulk workflow)
├── services/
│   └── xinno_whatsapp.py  # Xinno API client
├── utils/
│   ├── validators.py      # Columns, validation, normalization
│   ├── bulk_send.py       # Bulk eligibility & sequential send
│   └── audit.py           # Audit records, history, CSV export
├── tests/                 # Automated tests (mocked Xinno)
├── assets/                # UI assets
├── data/                  # Sample datasets
├── docs/                  # Xinno API reference material
├── logs/                  # Local send logs (gitignored)
├── .env.example
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Setup

```bash
cd ai-mediastra-whatsapp-reminder
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with real Xinno credentials (never commit `.env`):

```env
XINNO_API_URL=https://whatsapp.xinno.in/REST/directApi/message
XINNO_API_KEY=your_xinno_api_key_here
XINNO_WABA_NUMBER=919515473474
WHATSAPP_TEMPLATE_NAME=reminder_refill_followup_v3
WHATSAPP_TEMPLATE_LANGUAGE=en
MEDICAL_STORE_NAME=PHARMA HUBB
```

---

## Run the application

```bash
streamlit run app.py
```

---

## Testing

All automated tests mock Xinno. They do not send real WhatsApp messages.

```bash
pytest -v
```

Current suite: **154+ tests** covering validation, normalization, column aliases, bulk send, audit, and UI safety checks.

---

## Security

- Secrets live in `.env` (gitignored); `.env.example` has placeholders only  
- API keys are masked in logs and never stored in audit CSV exports  
- Phone numbers are masked in session history and downloads  
- No automatic retries or uncontrolled parallel sends  

---

## Limitations

- Bulk send is sequential (by design)  
- Session history is in-memory for the current Streamlit session (not a database)  
- API acceptance is not the same as handset delivery; check Xinno/Meta for delivery status  
- Streamlit Community Cloud needs secrets configured separately from local `.env`  
