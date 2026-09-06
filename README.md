# PHARMA HUBB WhatsApp Campaign Center

A production-grade Streamlit application for bulk WhatsApp medicine refill reminders, built for **AI Mediastra / PHARMA HUBB**.

This repository provides two dedicated, independent bulk messaging campaign applications:
1. **Text-Only Campaign** (`app.py`): Standard medicine refill reminders.
2. **Image + Text Campaign** (`app_image_campaign.py`): Rich media medicine refill reminders featuring dynamic customer medicine lists, store branch contacts, manager helplines, and an image header.

---

## Project Structure

```text
ai-mediastra-whatsapp-reminder/
├── app.py                          # Streamlit UI: Existing Text-Only Campaign
├── app_image_campaign.py           # Streamlit UI: Production Image + Text Campaign
├── services/
│   ├── xinno_whatsapp.py           # Xinno API Client for Text Campaign
│   ├── xinno_image_template.py     # Xinno API Client for Image + Text Campaign
│   └── cloudinary_image.py         # Cloudinary Image Upload & URL Generator
├── utils/
│   ├── column_aliases.py           # Centralized CSV Alias Resolution & Canonical Normalization
│   ├── image_campaign.py           # Grouping, Medicine List Building, Variable Generation
│   ├── validators.py               # Phone Normalization & Validation Logic
│   ├── bulk_send.py                # Sequential Bulk Sending Engine & Idempotency
│   └── audit.py                    # Masked Audit Logging & Export Helpers
├── tests/                          # 271+ Automated Unit & Integration Tests (100% Mocked)
├── assets/                         # Brand and UI Assets
├── data/                           # Sample Datasets for Testing & Dry-Runs
├── docs/                           # WhatsApp/Xinno API Specifications & Postman References
├── logs/                           # Local Audit Logs (gitignored)
├── scripts/                        # Dry-Run & Verification Runners
├── .env.example                    # Safe Environment Configuration Template
├── pytest.ini                      # Pytest Configuration
├── requirements.txt                # Python Dependencies
└── README.md                       # Complete Project Documentation
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Clone & Environment Setup
```bash
# Navigate to workspace
cd ai-mediastra-whatsapp-reminder

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell / Command Prompt):
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your credentials (never commit `.env`):

```bash
cp .env.example .env
```

Key environment variables in `.env`:
```env
# --- Xinno CPaaS Gateway ---
XINNO_API_URL=https://whatsapp.xinno.in/REST/directApi/message
XINNO_API_KEY=your_xinno_api_key_here
XINNO_WABA_NUMBER=919515473474
WHATSAPP_TEMPLATE_LANGUAGE=en

# --- Text Campaign ---
WHATSAPP_TEMPLATE_NAME=reminder_refill_followup_v3

# --- Image Campaign ---
XINNO_IMAGE_TEMPLATE_NAME=refill_reminder_image
XINNO_IMAGE_URL=https://res.cloudinary.com/troli5kq/image/upload/f_auto,q_auto/PHARMA_HUBB_-_medicine_Refill_Reminder_Image_Template

# --- Cloudinary (For uploading image files from device) ---
CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret

# --- Business Details ---
MEDICAL_STORE_NAME=PHARMA HUBB
```

---

## Local Run

### Launching the Applications
Each campaign runs independently on its own Streamlit server:

- **Run Image + Text Campaign (Production Refill Reminder)**:
  ```bash
  streamlit run app_image_campaign.py
  ```

- **Run Text-Only Campaign**:
  ```bash
  streamlit run app.py
  ```

---

## Flexible CSV Column Alias System

The Image Campaign features a production-grade CSV alias resolution engine (`utils/column_aliases.py`). Clients do **not** need to manually rename columns if they use common pharmacy or CRM headers.

### Supported Canonical Fields & Example Aliases

| Canonical Field | Internal Name | Accepted Header Variations (167+ aliases supported) | Required? |
|---|---|---|---|
| `customer_name` | `Name` | `Name`, `Customer Name`, `customer_name`, `Patient Name`, `Full Name`, `Client Name` | **Yes** |
| `phone` | `Phone number` | `Phone`, `Phone Number`, `phone_number`, `Mobile`, `Mobile Number`, `WhatsApp Number`, `Contact Number` | **Yes** |
| `medicine` | `Medicine` | `Medicine`, `Medicine Name`, `Medication`, `Drug`, `Product`, `Customer Medication List`, `Refill Medicine` | **Yes** |
| `branch` | `Branch` | `Branch`, `Branch Name`, `Location`, `Outlet`, `Store Branch`, `Branch / Location` | **Yes** |
| `store_name` | `Store Name` | `Store`, `Store Name`, `Pharmacy`, `Shop Name`, `Business Name` | Optional (falls back to `.env`) |
| `contact_no` | `Contact No.` | `Contact No.`, `Store Contact`, `Store Phone`, `Pharmacy Phone`, `Store Contact Number` | Optional |
| `manager_contact` | `Manager Contact` | `Manager Contact`, `Manager Phone`, `Manager Mobile`, `Manager Number`, `Manager Contact No` | Optional |

### Ambiguity & Safety Protections
- **Ambiguity Detection**: If two uploaded columns both alias to the same field (e.g. both `Mobile` and `WhatsApp Number`), the system flags an ambiguity error and will not guess.
- **Missing Required Fields**: If any required field (`customer_name`, `phone`, `medicine`, `branch`) is missing, execution is blocked with clear instructions and examples.
- **Unmapped Extra Columns**: Additional columns (e.g. `Doctor Name`, `Notes`) are safely ignored without causing validation failure.

---

## Customer Grouping & Medicine Aggregation

1. **Identity Grouping**: Rows are grouped by **(normalized customer name + normalized Indian mobile number)**.
2. **Multi-Row Aggregation**: When a customer has multiple medications across separate rows, all medications are combined into a single, clean **comma-separated single-line string** (e.g. `METFORMIN 500 MG, TELMISARTAN 40 MG, ATORVASTATIN 10 MG`).
3. **Parameter Compliance**: To comply with Meta WhatsApp Cloud API rules, all parameters are sanitized to remove newlines, carriage returns, tabs, and excessive whitespace.
4. **Data Conflict Detection**: If rows sharing the same customer identity contain conflicting Branch, Store Contact, or Manager Contact values, the customer is flagged as invalid rather than guessing.

---

## WhatsApp Image Template

- **Template Name**: `refill_reminder_image`
- **Language**: `en` (deterministic)
- **Header Component**: `image` containing a public HTTPS link (Cloudinary or custom HTTPS URL).
- **Body Component**: Exactly **8 parameters** in strict order:
  1. `{{1}}`: Customer Name
  2. `{{2}}`: Store Name (`PHARMA HUBB`)
  3. `{{3}}`: Store Branch (`Chadargatt`)
  4. `{{4}}`: Dynamic Medicine List (`METFORMIN 500 MG, TELMISARTAN 40 MG, ATORVASTATIN 10 MG`)
  5. `{{5}}`: Store Contact Number (`9581473474`)
  6. `{{6}}`: Manager Contact Number (`9885473474`)
  7. `{{7}}`: Store Name (`PHARMA HUBB`)
  8. `{{8}}`: Store Branch (`Chadargatt`)

---

## Campaign Modes & Safety Controls

### Dry-Run Mode (Default)
- **Default State**: Dry-Run is always enabled by default.
- **Zero Real Requests**: Constructs, inspects, and validates the exact JSON payloads without making HTTP calls to WhatsApp/Xinno.
- **Visual Confidence**: Visual preview shows full customer count, medicine counts, and variable substitutions.

### Live Send Mode
- Requires explicitly switching the toggle to **Live Send** and checking the confirmation checkbox.
- Sends sequentially to eligible customers with distinct attempt IDs.
- Double-click and page-rerun duplicate protection guarantees each recipient is messaged at most once.

---

## Audit Logging & Security

- **Masked Phone Numbers**: All audit records and log entries mask customer phone numbers (`91******5016`).
- **Zero Secrets Logged**: API keys, Cloudinary secrets, and authentication tokens are strictly stripped and masked in all log outputs.
- **Log Location**: Written to `logs/whatsapp_send.log` (gitignored).

---

## Automated Test Suite

The project includes **271 automated unit and integration tests** covering phone normalization, alias matching, grouping, medicine aggregation, template safety, dry-run simulation, and audit logging.

To run the full test suite:
```bash
python -m pytest ai-mediastra-whatsapp-reminder -v
```

All tests are 100% mocked and make **zero** real network or WhatsApp requests.

---

## Production Deployment

### Option A: Streamlit Community Cloud
1. Push the repository to a private GitHub repository.
2. In Streamlit Cloud, create a new app and set the Main file path to `app_image_campaign.py`.
3. Under **App Settings → Secrets**, add all variables from `.env.example`:
   ```toml
   XINNO_API_URL = "https://whatsapp.xinno.in/REST/directApi/message"
   XINNO_API_KEY = "your_actual_key"
   XINNO_WABA_NUMBER = "919515473474"
   XINNO_IMAGE_TEMPLATE_NAME = "refill_reminder_image"
   XINNO_IMAGE_URL = "https://res.cloudinary.com/..."
   CLOUDINARY_CLOUD_NAME = "your_cloud_name"
   CLOUDINARY_API_KEY = "your_api_key"
   CLOUDINARY_API_SECRET = "your_api_secret"
   MEDICAL_STORE_NAME = "PHARMA HUBB"
   ```

### Option B: Linux VPS / Docker / Cloud Run
1. Deploy as a container or systemd service using:
   ```bash
   streamlit run app_image_campaign.py --server.port 8501 --server.address 0.0.0.0
   ```
2. Configure environment variables via your container management or `.env` file.
