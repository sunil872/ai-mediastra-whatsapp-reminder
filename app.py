"""
AI Mediastra WhatsApp Reminder
Streamlit application entry point.

Phase 2: Customer CSV/XLSX upload and validation.
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from utils.validators import (
    check_required_columns,
    validate_customers,
    generate_message,
)
from services.xinno_whatsapp import send_template_message, get_config_diagnostic

# Load environment variables from .env file at project root
PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Mediastra - Medicine Refill Reminder",
    page_icon="💊",
    layout="centered",
)

# --- App Header ---
st.title("💊 AI Mediastra - Medicine Refill Reminder")
st.markdown("Send medicine refill reminders to your customers via WhatsApp.")

# --- Safe Configuration Diagnostic ---
with st.expander("⚙️ Environment Configuration Status", expanded=False):
    config_diag = get_config_diagnostic()
    for key, status in config_diag.items():
        icon = "✅" if status == "configured" else "❌"
        st.markdown(f"{icon} **{key}**: `{status}`")

st.divider()

# --- File Upload ---
st.subheader("📁 Upload Customer List")
uploaded_file = st.file_uploader(
    "Upload a CSV or XLSX file with **Name** and **Phone number** columns.",
    type=["csv", "xlsx"],
)

if uploaded_file is not None:
    # --- Read File ---
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str)
        else:
            df = pd.read_excel(uploaded_file, dtype=str)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

    # --- Column Validation ---
    missing = check_required_columns(df)
    if missing:
        st.error(
            f"❌ Missing required column(s): **{', '.join(missing)}**. "
            f"Your file has columns: {list(df.columns)}"
        )
        st.stop()

    # --- Validate Records ---
    valid_df, invalid_df, duplicate_df = validate_customers(df)

    total_records = len(df)
    valid_count = len(valid_df)
    invalid_count = len(invalid_df)
    duplicate_count = len(duplicate_df)

    # --- Summary Metrics ---
    st.divider()
    st.subheader("📊 Validation Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", total_records)
    col2.metric("Valid ✅", valid_count)
    col3.metric("Invalid ❌", invalid_count)
    col4.metric("Duplicates ⚠️", duplicate_count)

    # --- Valid Records ---
    st.divider()
    st.subheader("✅ Valid Customers")
    if valid_df.empty:
        st.warning("No valid customer records found.")
    else:
        st.dataframe(valid_df, use_container_width=True, hide_index=True)

    # --- Invalid Records ---
    if not invalid_df.empty:
        st.divider()
        st.subheader("❌ Invalid Records")
        st.dataframe(invalid_df, use_container_width=True, hide_index=True)

    # --- Duplicate Records ---
    if not duplicate_df.empty:
        st.divider()
        st.subheader("⚠️ Duplicate Records (by Phone Number)")
        st.dataframe(duplicate_df, use_container_width=True, hide_index=True)

    # --- Medical Store Name & Message Preview ---
    if not valid_df.empty:
        st.divider()
        st.subheader("🏪 Store Details")
        store_name = st.text_input("Medical Store Name", value=os.getenv("MEDICAL_STORE_NAME", "PHARMA HUBB").strip() or "PHARMA HUBB")

        st.divider()
        st.subheader("💬 Message Preview")

        customer_names = valid_df["Name"].tolist()
        selected_customer = st.selectbox(
            "Select a customer to preview the message:",
            options=customer_names,
        )

        if selected_customer and store_name:
            preview = generate_message(selected_customer, store_name)
            st.text_area(
                "Personalized Message",
                value=preview,
                height=250,
                disabled=True,
            )

    # --- WhatsApp API DRY RUN Test ---
    st.divider()
    st.subheader("📤 WhatsApp API (DRY RUN Test)")
    st.info(
        "🧪 **Phase 3 DRY RUN Mode**: Test constructing the Xinno API request payload. "
        "No actual WhatsApp messages will be sent."
    )

    if not valid_df.empty:
        if st.button("🧪 Run DRY RUN Test for Selected Customer"):
            selected_row = valid_df[valid_df["Name"] == selected_customer].iloc[0]
            phone_num = selected_row["Phone number"]
            dry_run_res = send_template_message(
                phone_number=phone_num,
                customer_name=selected_customer,
                store_name=store_name,
                dry_run=True
            )
            st.success("✅ DRY RUN payload constructed successfully! (No HTTP request sent)")
            st.json(dry_run_res)

    # --- Phase 4: Controlled Live Test (Hard-Locked to Sunil) ---
    st.divider()
    st.subheader("🚨 Phase 4: Controlled Live Test (Single Message Only)")
    st.warning(
        "⚠️ **LIVE TEST MODE**: This section allows sending **ONE single real WhatsApp message** "
        "hard-locked to the designated test customer **Sunil**. Bulk sending remains strictly disabled."
    )

    # Hard-locked test parameters for safety
    customer_name = "Sunil"
    phone_number = "7659935016"
    store_name = "PHARMA HUBB"
    normalized_phone = "917659935016"
    template_name = "reminder_refill_followup_v2"

    st.write(f"Customer: {customer_name}")
    st.write(f"Phone: {phone_number}")
    st.write(f"Normalized Phone: {normalized_phone}")
    st.write(f"Store: {store_name}")
    st.write(f"Template: {template_name}")

    confirm_live = st.checkbox(
        "I understand this will send ONE real WhatsApp message to Sunil at 7659935016."
    )

    if st.button("🚀 Send 1 Real Live Test Message to Sunil", disabled=not confirm_live):
        with st.spinner("Sending single live message to Sunil via Xinno WhatsApp API..."):
            live_res = send_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                dry_run=False
            )
        if live_res["success"]:
            st.success(f"✅ {live_res['message']}")
        else:
            st.error(f"❌ {live_res['message']}")
        st.json(live_res)

    st.divider()
    st.button(
        "Send Bulk Reminders via WhatsApp (Disabled)",
        disabled=True,
        help="Bulk sending remains disabled in Phase 4.",
    )
else:
    st.info("👆 Upload a CSV or XLSX file to get started.")
