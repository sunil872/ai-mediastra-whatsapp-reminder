# AI Mediastra — Bulk WhatsApp Medicine Refill Reminder

A production-ready **bulk WhatsApp messaging** application for pharmacy medicine refill follow-ups.

Business: **AI Mediastra / PHARMA HUBB**  
Channel: **Xinno WhatsApp CPaaS**  
Template: **`reminder_refill_followup_v3`** (language `en`)

---

## Purpose

Operators upload a customer list (CSV or Excel). The system validates and normalizes phones, shows a bulk preview, and — only after explicit confirmation — sends the approved refill template to **all eligible (Valid) customers**, sequentially. Every attempt is audited and exportable as CSV.

This is a **bulk messaging system only**. There is no single-customer send workflow.

---

## Final workflow

```
Upload customer file (CSV / XLSX)
        ↓
Validate customer data
        ↓
Normalize phone numbers
        ↓
Identify Valid / Invalid / Duplicate
        ↓
Bulk send preview + eligible recipient count
        ↓
Explicit bulk send confirmation
        ↓
Send to all Valid customers (sequential)
        ↓
Track each message result
        ↓
Bulk send summary
        ↓
Download audit / results CSV
```

Upload and validation **never** send messages automatically. Real sends use `dry_run=False` only on the confirmed bulk-send button.

---

## Key features

1. **Bulk-only WhatsApp sending** — all Valid customers in one confirmed operation  
2. **Validation & deduplication** — missing data, invalid mobiles, duplicates after normalization  
3. **Indian phone normalization** → `91XXXXXXXXXX`  
4. **Dynamic personalization** — `{{1}}` = each customer’s Name; `{{2}}`/`{{3}}` = PHARMA HUBB  
5. **Explicit confirmation** before any live send  
6. **Sequential sending** — continue on failure; no automatic retries  
7. **Bulk attempt ID** — protects against Streamlit reruns / double-clicks  
8. **Per-message audit** + masked CSV export  
9. **Secrets stay in `.env`** — never shown in UI, logs, or exports  

---

## Input file format

### Supported
- CSV (`.csv`)
- Excel (`.xlsx`)

### Required columns
| Column | Description |
|--------|-------------|
| `Name` | Customer full name |
| `Phone number` | Indian mobile in common formats |

Extra columns are allowed and ignored for messaging.

### Example phone formats (all normalize to `91…`)
- `7659935016`
- `+91 76599 35016`
- `+917659935016`
- `9176599 35016`
- `917659935016`

Duplicate detection runs **after** normalization. Only `Status = Valid` customers are eligible.

---

## WhatsApp template (do not change)

| Setting | Value |
|---------|--------|
| Template | `reminder_refill_followup_v3` |
| Language | `en` (policy: deterministic) |
| Pharmacy | `PHARMA HUBB` |
| `{{1}}` | Customer Name (from file) |
| `{{2}}` | PHARMA HUBB |
| `{{3}}` | PHARMA HUBB |

---

## Xinno configuration

Keep credentials in `.env` (never commit secrets):

```
XINNO_API_URL=https://whatsapp.xinno.in/REST/directApi/message
XINNO_API_KEY=your_xinno_api_key_here
XINNO_WABA_NUMBER=919515473474
WHATSAPP_TEMPLATE_NAME=reminder_refill_followup_v3
WHATSAPP_TEMPLATE_LANGUAGE=en
MEDICAL_STORE_NAME=PHARMA HUBB
```

Copy from `.env.example`. The app reuses `services/xinno_whatsapp.py` (`send_template_message`). Default `dry_run=True` except the confirmed bulk path.

---

## Running the application

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Create .env from .env.example and set real credentials
streamlit run app.py
```

---

## Testing

All automated tests **mock** Xinno. They do **not** send real WhatsApp messages.

```bash
pytest -v
# or explicitly:
pytest tests -v
```

Coverage includes validation, normalization, duplicates, bulk payloads, failure isolation, confirmation, attempt-ID protection, audit/export, masking, and Streamlit bulk-only UI checks.

---

## Project layout

```
ai-mediastra-whatsapp-reminder/
├── app.py                 # Bulk-only Streamlit UI
├── pytest.ini             # Pytest config (tests/ + project path)
├── requirements.txt
├── .env.example
├── README.md
├── assets/                # UI assets (e.g. WhatsApp icon)
├── data/                  # Sample CSV / XLSX datasets
├── docs/                  # API reference docs
├── logs/                  # Local send logs (gitignored)
├── services/
│   └── xinno_whatsapp.py  # Xinno REST client
├── utils/
│   ├── validators.py      # Validation & normalization
│   ├── bulk_send.py       # Bulk send helpers
│   └── audit.py           # Audit records & CSV export
└── tests/                 # All automated tests (mocked Xinno)
    ├── conftest.py
    ├── test_bulk_send.py
    ├── test_dynamic_preview.py
    ├── test_phase5_validation.py
    ├── test_phase7_audit.py
    ├── test_phase8_prelive.py
    ├── test_phone_normalization.py
    └── test_xinno.py
```

---

## Security

- `.env` is gitignored; `.env.example` has placeholders only  
- API keys never appear in UI, logs, audit, or CSV  
- Phones are masked in results and exports (e.g. `919******688`)  
- No automatic retries; no parallel/unbounded send fan-out  

---

## Important notes

- **API “Accepted” ≠ phone delivery.** Meta may still drop marketing messages later (e.g. per-user frequency limits).  
- Do not hard-code customer names or phones in the production workflow — all data comes from the uploaded file.  
`