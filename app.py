"""
AI Mediastra WhatsApp Reminder
Bulk WhatsApp medicine refill reminder system (Streamlit).

Workflow: Upload → Validate → Normalize → Preview → Confirm → Bulk Send → Audit → Export

Real messages require explicit operator confirmation. dry_run=False only on that path.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.validators import (
    check_required_columns,
    missing_columns_message,
    normalize_columns,
    validate_customers,
    build_preview_table,
    WHATSAPP_TEMPLATE_NAME,
    NORMALIZED_PHONE_LABEL,
)
from utils.audit import (
    append_send_history,
    history_to_dataframe,
    history_to_csv_bytes,
    log_audit_record_safely,
)
from utils.bulk_send import (
    get_eligible_customers,
    build_bulk_summary,
    build_sample_message_previews,
    bulk_confirmation_ready,
    new_bulk_attempt_id,
    execute_bulk_send,
    bulk_results_table,
)
from services.xinno_whatsapp import send_template_message

PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
LOG_FILE = PROJECT_ROOT / "logs" / "whatsapp_send.log"
WHATSAPP_ICON = PROJECT_ROOT / "assets" / "whatsapp.svg"
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

st.set_page_config(
    page_title="AI Mediastra — WhatsApp Reminder",
    page_icon="💬",
    layout="centered",
)

# Session audit history
if "send_history" not in st.session_state:
    st.session_state.send_history = []
if "phase7_recorded_attempt_ids" not in st.session_state:
    st.session_state.phase7_recorded_attempt_ids = set()

# Bulk operation locks
if "bulk_send_in_progress" not in st.session_state:
    st.session_state.bulk_send_in_progress = False
if "bulk_completed_attempt_ids" not in st.session_state:
    st.session_state.bulk_completed_attempt_ids = set()
if "bulk_started_attempt_ids" not in st.session_state:
    st.session_state.bulk_started_attempt_ids = set()
if "bulk_active_attempt_id" not in st.session_state:
    st.session_state.bulk_active_attempt_id = None
if "bulk_last_summary" not in st.session_state:
    st.session_state.bulk_last_summary = None


def _whatsapp_icon_data_uri() -> str:
    if not WHATSAPP_ICON.exists():
        return ""
    raw = WHATSAPP_ICON.read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


# --- Header ---
st.title("💊 Mediastra WhatsApp Reminder")
st.caption(
    "Medicine refill reminders for PHARMA HUBB via WhatsApp. "
    "Upload a customer list, review validation, confirm, then send to all eligible customers. "
    "Messages are never sent automatically on upload."
)

# --- Upload ---
st.subheader("Upload customer file")
uploaded_file = st.file_uploader(
    "Upload a CSV or XLSX customer list",
    type=["csv", "xlsx"],
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str)
        else:
            df = pd.read_excel(uploaded_file, dtype=str)
    except Exception:
        st.error("Failed to read file. Please upload a valid CSV or XLSX file.")
        st.stop()

    df = normalize_columns(df)
    missing = check_required_columns(df)
    if missing:
        st.error(missing_columns_message(missing))
        st.caption(f"Your file has columns: {list(df.columns)}")
        st.stop()

    valid_df, invalid_df, duplicate_df = validate_customers(df)

    total_records = len(df)
    valid_count = len(valid_df)
    invalid_count = len(invalid_df)
    duplicate_count = len(duplicate_df)
    eligible_customers = get_eligible_customers(valid_df)
    eligible_count = len(eligible_customers)

    store_name = os.getenv("MEDICAL_STORE_NAME", "PHARMA HUBB").strip() or "PHARMA HUBB"
    template_name = (
        os.getenv("WHATSAPP_TEMPLATE_NAME", WHATSAPP_TEMPLATE_NAME).strip()
        or WHATSAPP_TEMPLATE_NAME
    )
    template_language = os.getenv("WHATSAPP_TEMPLATE_LANGUAGE", "en").strip() or "en"

    # --- Validation summary ---
    st.subheader("Validation summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", total_records)
    col2.metric("Valid", valid_count)
    col3.metric("Invalid", invalid_count)
    col4.metric("Duplicates", duplicate_count)

    # --- Customer data preview ---
    st.subheader("Customer data preview")
    st.caption(
        "Phone numbers are normalized to WhatsApp format 91XXXXXXXXXX. "
        "Duplicates are detected after normalization."
    )
    preview_df = build_preview_table(valid_df, invalid_df, duplicate_df)
    if preview_df.empty:
        st.warning("No customer records found in the uploaded file.")
    else:
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    if not invalid_df.empty:
        with st.expander("Invalid records", expanded=False):
            st.dataframe(
                invalid_df[["Row", "Name", "Original Phone", "Reason", "Status"]],
                use_container_width=True,
                hide_index=True,
            )

    if not duplicate_df.empty:
        with st.expander("Duplicate phone numbers (excluded from send)", expanded=False):
            st.dataframe(
                duplicate_df[["Name", "Original Phone", "Normalized Phone", "Status"]].rename(
                    columns={"Normalized Phone": NORMALIZED_PHONE_LABEL}
                ),
                use_container_width=True,
                hide_index=True,
            )

    # --- Bulk messaging ---
    st.subheader("Bulk WhatsApp messaging")

    store_name = st.text_input("Pharmacy name", value=store_name)

    summary = build_bulk_summary(
        total_rows=total_records,
        valid_count=valid_count,
        invalid_count=invalid_count,
        duplicate_count=duplicate_count,
        eligible_count=eligible_count,
        template_name=template_name,
        template_language=template_language,
        pharmacy_name=store_name,
    )

    st.markdown("##### Send preview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Eligible recipients", summary["eligible_for_sending"])
    m2.write(f"**Template:** {summary['template_name']}")
    m3.write(f"**Language:** {summary['template_language']}")
    st.write(f"**Pharmacy:** {summary['pharmacy_name']}")
    st.write(
        f"Uploaded: {summary['total_uploaded_rows']} · "
        f"Valid: {summary['valid_customers']} · "
        f"Invalid: {summary['invalid_customers']} · "
        f"Duplicates: {summary['duplicate_customers']}"
    )

    if eligible_count == 0:
        st.warning("No valid customers are available for sending.")
    else:
        eligible_display = pd.DataFrame(eligible_customers)[
            ["Name", "Original Phone", "Normalized Phone", "Status"]
        ].rename(columns={
            "Name": "Customer Name",
            "Normalized Phone": NORMALIZED_PHONE_LABEL,
        })
        st.dataframe(eligible_display, use_container_width=True, hide_index=True)

        st.markdown("##### Message preview")
        st.caption("Samples use each customer's name from the uploaded file.")
        samples = build_sample_message_previews(
            eligible_customers, store_name, limit=min(3, eligible_count)
        )
        for sample in samples:
            with st.expander(
                f"{sample['customer_name']} · {sample['normalized_phone']}"
            ):
                st.code(
                    f"{{{{1}}}} = {sample['var1']}\n"
                    f"{{{{2}}}} = {sample['var2']}\n"
                    f"{{{{3}}}} = {sample['var3']}",
                    language=None,
                )
                st.text_area(
                    "Message preview",
                    value=sample["message_preview"],
                    height=200,
                    disabled=True,
                    key=f"bulk_sample_{sample['normalized_phone']}_{sample['customer_name']}",
                )

        st.warning(
            f"**Confirm before sending**\n\n"
            f"Recipients: **{eligible_count}**\n\n"
            "This will send WhatsApp messages to all eligible customers listed above."
        )

        understand_bulk = st.checkbox(
            "I understand that this will send real WhatsApp messages to all eligible customers shown above.",
            key="bulk_understand",
            disabled=st.session_state.bulk_send_in_progress,
        )

        bulk_ready = (
            bulk_confirmation_ready(understand_bulk)
            and eligible_count > 0
            and not st.session_state.bulk_send_in_progress
        )

        # WhatsApp icon + primary send button
        wa_uri = _whatsapp_icon_data_uri()
        icon_col, btn_col = st.columns([0.12, 0.88])
        with icon_col:
            if wa_uri:
                st.markdown(
                    f'<img src="{wa_uri}" width="40" height="40" '
                    f'alt="WhatsApp" style="margin-top:4px;" />',
                    unsafe_allow_html=True,
                )
        with btn_col:
            send_clicked = st.button(
                "Send WhatsApp Messages",
                disabled=not bulk_ready,
                key="bulk_send_btn",
                type="primary",
                use_container_width=True,
            )

        if send_clicked:
            if not st.session_state.bulk_active_attempt_id:
                st.session_state.bulk_active_attempt_id = new_bulk_attempt_id()
            bulk_id = st.session_state.bulk_active_attempt_id

            if (
                bulk_id in st.session_state.bulk_completed_attempt_ids
                or bulk_id in st.session_state.bulk_started_attempt_ids
            ):
                st.error(
                    "This bulk attempt was already started or completed. "
                    "Reload the page to start a new bulk send."
                )
            elif not bulk_confirmation_ready(understand_bulk):
                st.error("Confirmation is required before bulk sending.")
            elif eligible_count == 0:
                st.error("No valid customers are available for sending.")
            else:
                st.session_state.bulk_started_attempt_ids.add(bulk_id)
                st.session_state.bulk_send_in_progress = True
                try:
                    progress = st.progress(0.0, text="Sending messages...")
                    status_box = st.empty()

                    def _send_one(phone_number, customer_name, store_name, dry_run):
                        return send_template_message(
                            phone_number=phone_number,
                            customer_name=customer_name,
                            store_name=store_name,
                            dry_run=dry_run,
                        )

                    summary_result = execute_bulk_send(
                        eligible_customers,
                        store_name=store_name,
                        template_name=template_name,
                        template_language=template_language,
                        send_fn=_send_one,
                        dry_run=False,
                        bulk_attempt_id=bulk_id,
                    )

                    for rec in summary_result["records"]:
                        aid = rec["attempt_id"]
                        if aid not in st.session_state.phase7_recorded_attempt_ids:
                            st.session_state.send_history = append_send_history(
                                st.session_state.send_history, rec
                            )
                            st.session_state.phase7_recorded_attempt_ids.add(aid)
                            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                            log_audit_record_safely(rec, str(LOG_FILE))

                    progress.progress(1.0, text="Send finished.")
                    st.session_state.bulk_completed_attempt_ids.add(bulk_id)
                    st.session_state.bulk_last_summary = summary_result
                    st.session_state.bulk_active_attempt_id = None
                    status_box.success("Bulk send completed.")
                finally:
                    st.session_state.bulk_send_in_progress = False


    # --- Bulk results ---
    if st.session_state.bulk_last_summary:
        st.subheader("Send results")
        s = st.session_state.bulk_last_summary
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Eligible", s["eligible"])
        c2.metric("Attempted", s["attempted"])
        c3.metric("Accepted", s["successful"])
        c4.metric("Failed", s["failed"])
        c5.metric("Skipped", s["skipped"])
        results_df = bulk_results_table(s["records"], mask_phones=True)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        if s["records"]:
            st.download_button(
                label="Download send results",
                data=history_to_csv_bytes(s["records"], mask_phones=True),
                file_name=f"bulk_send_results_{s['bulk_attempt_id'][:8]}.csv",
                mime="text/csv",
                key="download_bulk_results",
            )

    st.subheader("Session history")
    st.caption("Current session only. Phone numbers are masked in the table and CSV export.")
    history_df = history_to_dataframe(st.session_state.send_history, mask_phones=True)
    if history_df.empty:
        st.info("No send attempts recorded in this session yet.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv_bytes = history_to_csv_bytes(st.session_state.send_history, mask_phones=True)
        st.download_button(
            label="Download session history",
            data=csv_bytes,
            file_name="session_send_history.csv",
            mime="text/csv",
            key="download_send_history",
        )
else:
    st.info("Upload a CSV or XLSX file to get started.")
    st.subheader("Session history")
    st.caption("Current session only.")
    history_df = history_to_dataframe(st.session_state.send_history, mask_phones=True)
    if history_df.empty:
        st.info("No send attempts recorded in this session yet.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv_bytes = history_to_csv_bytes(st.session_state.send_history, mask_phones=True)
        st.download_button(
            label="Download session history",
            data=csv_bytes,
            file_name="session_send_history.csv",
            mime="text/csv",
            key="download_send_history_empty",
        )
