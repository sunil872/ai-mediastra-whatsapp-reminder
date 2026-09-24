# WhatsApp Standalone Campaign Tools

This directory contains standalone, ad-hoc WhatsApp broadcast campaign applications for **AI Mediastra / PHARMA HUBB**.

These tools operate independently from the automated **RefillCare Intelligence Platform**. Use these apps when an operator wants to manually upload an ad-hoc customer CSV list and perform a controlled batch broadcast.

---

## Campaign Applications

### 1. Text-Only Campaign (`app.py`)
Standard medicine refill reminder messages based on pre-approved WhatsApp templates (`reminder_refill_followup_v3`).

* **Launch command:**
  ```bash
  streamlit run whatsapp_campaigns/app.py
  ```
* **Workflow:**
  1. Upload Customer CSV (e.g. from `whatsapp_campaigns/sample_data/sample_customers.csv`).
  2. Automatic Indian Phone Number Validation & Canonical E.164 Normalization.
  3. Dynamic Message Previews & Safety Verification.
  4. Explicit Operator Confirmation (Default: Strict DRY-RUN simulation).
  5. Bulk Send & Masked Audit Logging.

### 2. Image + Text Campaign (`app_image_campaign.py`)
Rich-media broadcast campaign featuring an image header, dynamic medicine list grouping, store branch contact details, and manager helplines.

* **Launch command:**
  ```bash
  streamlit run whatsapp_campaigns/app_image_campaign.py
  ```
* **Workflow:**
  1. Upload Customer CSV (e.g. from `whatsapp_campaigns/sample_data/image_campaign_sample.csv`).
  2. Cloudinary-hosted Campaign Banner Image integration.
  3. Customer Multi-Medicine aggregation into formatted WhatsApp bullets.
  4. Real-time visual message preview.
  5. Operator Confirmation & Controlled Dispatch.

---

## Sample Data Directory (`sample_data/`)

* `sample_customers.csv`: Minimal test dataset for the text campaign.
* `phase8_sample_customers.csv`: Real-world edge-case phone number test set.
* `image_campaign_sample.csv`: Sample customer rows with multiple medicine lines per patient.
* `image_campaign_single_test_customer.csv`: Single-row test dataset for live canary testing.
