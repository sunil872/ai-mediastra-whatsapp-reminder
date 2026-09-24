# Approved RefillCare WhatsApp Reminder Template Specification

## 1. Template Overview & Approved Metadata

The RefillCare Medication Refill Reminder System integrates with Meta/Xinno WhatsApp Business API using the following approved business template:

| Specification Field | Approved Value |
| :--- | :--- |
| **Template Name** | `refillcare_medicine_reminder` |
| **Category** | `Utility` |
| **Type** | `text` |
| **Language** | `en` |
| **Language Policy** | `deterministic` |
| **Total Variables** | `6` |

---

## 2. Exact Approved Template Body Text

```text
Dear *{{1}}*,

This is a friendly reminder from *{{2}}* regarding your regular medicine refill.

{{3}}

💊 Medicines:
{{4}}

If you need a refill, please contact us or place your order with our pharmacy.

📞 Contact: {{5}}

Thank you for choosing *{{6}}*.
🙏 We are happy to serve you.
```

---

## 3. Exact 6-Variable Order Mapping

The six variables are **FIXED** and must strictly appear in this exact order:

| Position | Variable Name | Description | Source in RefillCare | Example Value |
| :---: | :--- | :--- | :--- | :--- |
| `{{1}}` | **Customer Name** | Patient / Customer sanitized name | `customerName` (or `customerId`) | `John Doe` |
| `{{2}}` | **Store Name** | Pharmacy name | `MEDICAL_STORE_NAME` or UI input | `PHARMA HUBB` |
| `{{3}}` | **Reminder-Stage Message** | Clinical reminder copy from Phase 5 scheduler | `reminder.scheduler.format_reminder_message()` | `Your regular medicine refill may be due in about 7 days, around 15-09-2026.` |
| `{{4}}` | **Medicine Name / List** | Prescribed medication item(s) | `itemName` (or `itemId`) | `TELMISARTAN 40MG` |
| `{{5}}` | **Store Contact** | Pharmacy phone number for orders | `STORE_CONTACT_NUMBER` or UI input | `9876543210` |
| `{{6}}` | **Store Name** | Pharmacy name in closing greeting | `MEDICAL_STORE_NAME` or UI input | `PHARMA HUBB` |

> [!IMPORTANT]
> `{{2}}` and `{{6}}` intentionally both contain the Store Name. `{{2}}` appears in the introductory greeting, while `{{6}}` appears in the closing thank-you sentence (*"Thank you for choosing \*{{6}}\*."*). Do **not** remove or alter `{{6}}`.

---

## 4. Approved JSON Payload Structure

The reference payload submitted to the WhatsApp Direct API endpoint (`POST /REST/directApi/message`):

```json
{
  "messaging_product": "whatsapp",
  "to": "919849012345",
  "type": "template",
  "template": {
    "name": "refillcare_medicine_reminder",
    "language": {
      "code": "en",
      "policy": "deterministic"
    },
    "components": [
      {
        "type": "BODY",
        "parameters": [
          {
            "type": "text",
            "text": "John Doe"
          },
          {
            "type": "text",
            "text": "PHARMA HUBB"
          },
          {
            "type": "text",
            "text": "Your regular medicine refill may be due in about 7 days, around 15-09-2026."
          },
          {
            "type": "text",
            "text": "TELMISARTAN 40MG"
          },
          {
            "type": "text",
            "text": "9876543210"
          },
          {
            "type": "text",
            "text": "PHARMA HUBB"
          }
        ]
      }
    ]
  }
}
```

---

## 5. Dynamic Reminder-Stage Messages for Variable `{{3}}`

Variable `{{3}}` is dynamically determined by the **Phase 5 Reminder Scheduling Engine** based on the stage offset relative to the predicted `expected_refill_date` ($D_{\text{exp}}$):

| Stage Offset | Name | Approved Deterministic Text Template for `{{3}}` |
| :---: | :--- | :--- |
| **`-7`** | 7-Day Early Notice | `Your regular medicine refill may be due in about 7 days, around DD-MM-YYYY.` |
| **`-3`** | 3-Day Upcoming | `Your regular medicine refill may be due in about 3 days, around DD-MM-YYYY.` |
| **`-1`** | 1-Day Urgent | `Your regular medicine refill may be due tomorrow, DD-MM-YYYY.` |
| **`0`** | Due Date | `Your regular medicine refill may be due today, DD-MM-YYYY.` |
| **`+2`** | +2 Day Follow-up | `Your expected refill date was DD-MM-YYYY. If you still need your medicine, please contact us.` |
| **`+5`** | +5 Day Overdue | `We wanted to follow up regarding your medicine refill. If you still need your medicine, please contact us.` |

> [!NOTE]
> All dates in `{{3}}` are formatted as `DD-MM-YYYY` (e.g. `15-09-2026`).

---

## 6. Operational Restrictions & Safety Guidelines

1. **No LLM Text Generation:** All variable content is generated strictly through deterministic string formatters to prevent hallucinated medical claims or template rejections.
2. **Control Character Sanitization:** All text variables are stripped of invalid newlines (`\n`, `\r`) and tab characters (`\t`) before insertion into payload parameters.
3. **No Credential Leakage:** Payloads logged or stored in the SQLite audit trail redact phone numbers to last 4 digits (`***2345`) and omit all API keys or authorization headers.
4. **Frozen Campaign Separation:** This configuration is isolated to the RefillCare subsystem (`refillcare/config/whatsapp_template.py`) and leaves legacy campaign applications (`app.py`, `app_image_campaign.py`) untouched.
