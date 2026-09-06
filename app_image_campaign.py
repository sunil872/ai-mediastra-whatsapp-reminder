"""
PHARMA HUBB — WhatsApp Refill Reminder (Image Campaign)
Client-facing Streamlit app for bulk image + text refill reminders.

Workflow: Upload Customers → Campaign Image → Review Message → Confirm → Send → Results

Separate Streamlit entry point — does NOT modify app.py or the text campaign.

    Run: streamlit run app_image_campaign.py
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.column_aliases import normalize_dataframe_columns
from utils.image_campaign import (
    check_image_campaign_columns,
    missing_image_columns_message,
    validate_and_group_customers,
    build_image_sample_previews,
    validate_image_url,
)
from services.cloudinary_image import (
    upload_image_to_cloudinary,
    validate_image_file,
)
from utils.audit import (
    append_send_history,
    history_to_csv_bytes,
    history_to_dataframe,
    log_audit_record_safely,
    mask_phone_for_audit,
)
from utils.bulk_send import (
    bulk_confirmation_ready,
    new_bulk_attempt_id,
    execute_bulk_send,
)
from services.xinno_image_template import send_image_template_message

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
LOG_FILE = PROJECT_ROOT / "logs" / "whatsapp_send.log"
WHATSAPP_ICON = PROJECT_ROOT / "assets" / "whatsapp.svg"
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

st.set_page_config(
    page_title="Mediastra WhatsApp Reminder — Image + Text Campaign",
    page_icon="💊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_SESSION_DEFAULTS = {
    "img_send_history": [],
    "img_recorded_attempt_ids": set(),
    "img_bulk_send_in_progress": False,
    "img_bulk_completed_attempt_ids": set(),
    "img_bulk_started_attempt_ids": set(),
    "img_bulk_active_attempt_id": None,
    "img_bulk_last_summary": None,
    "img_uploaded_cloudinary_url": None,
    "img_uploaded_filename": None,
    "img_show_confirm": False,
    "img_campaign_complete": False,
}
for _key, _default in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


def _whatsapp_icon_data_uri() -> str:
    if not WHATSAPP_ICON.exists():
        return ""
    raw = WHATSAPP_ICON.read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


def _client_customer_table(grouped_customers: List[Dict[str, Any]]) -> pd.DataFrame:
    """Simple customer readiness table for the client UI."""
    rows = []
    for c in grouped_customers:
        rows.append({
            "Customer": c["Name"],
            "Phone": c.get("Original Phone") or mask_phone_for_audit(c["Normalized Phone"]),
            "Branch": c.get("Branch", ""),
            "Medicines": c.get("Medicine Count", 0),
            "Status": "Ready",
        })
    return pd.DataFrame(rows)


def _client_attention_table(invalid_df: pd.DataFrame) -> pd.DataFrame:
    """Simple table for records that need attention."""
    if invalid_df is None or invalid_df.empty:
        return pd.DataFrame()
    display_cols = [c for c in ["Name", "Original Phone", "Medicine", "Reason"] if c in invalid_df.columns]
    out = invalid_df[display_cols].copy()
    rename = {"Name": "Customer", "Original Phone": "Phone"}
    return out.rename(columns={k: v for k, v in rename.items() if k in out.columns})


def _client_results_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Client-friendly send results (no API/template internals)."""
    rows = []
    for r in records:
        rows.append({
            "Customer": r.get("customer_name", ""),
            "Phone": mask_phone_for_audit(r.get("normalized_phone", "")),
            "Result": "Sent" if r.get("success") else "Failed",
            "Details": (r.get("error") or "") if not r.get("success") else "",
        })
    return pd.DataFrame(rows, columns=["Customer", "Phone", "Result", "Details"])


def _reset_campaign_session() -> None:
    """Clear campaign UI state so the client can start fresh."""
    st.session_state.img_bulk_last_summary = None
    st.session_state.img_bulk_active_attempt_id = None
    st.session_state.img_bulk_send_in_progress = False
    st.session_state.img_show_confirm = False
    st.session_state.img_campaign_complete = False
    st.session_state.img_uploaded_cloudinary_url = None
    st.session_state.img_uploaded_filename = None
    # Allow a new bulk attempt after completion
    st.session_state.img_bulk_completed_attempt_ids = set()
    st.session_state.img_bulk_started_attempt_ids = set()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Mediastra WhatsApp Reminder")
st.subheader("Image + Text Campaign")
st.write(
    "WhatsApp Image + Text medicine refill campaign for PHARMA HUBB. "
    "Upload a customer list with medicines, review grouped customers and "
    "template variables, confirm, then send to all eligible customers. "
    "Messages are never sent automatically on upload."
)

# ---------------------------------------------------------------------------
# Upload customer file
# ---------------------------------------------------------------------------
st.subheader("Upload customer file")
st.caption(
    "The file must contain columns: **Name**, **Phone number**, **Medicine**, "
    "**Branch**, **Contact No.**, **Manager Contact**.  "
    "The same customer can appear on multiple rows (one medicine per row)."
)
uploaded_file = st.file_uploader(
    "Upload a CSV or XLSX customer list",
    type=["csv", "xlsx"],
    key="img_file_uploader",
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file, dtype=str)
        else:
            raw_df = pd.read_excel(uploaded_file, dtype=str)
    except Exception as read_err:
        st.error(f"Could not read this file. Please upload a valid CSV or Excel file. ({read_err})")
        st.stop()

    # Alias resolution runs internally — never shown to the client
    df, alias_result = normalize_dataframe_columns(raw_df)

    if not alias_result.is_valid:
        friendly_col_names = {
            "customer_name": "Name",
            "phone": "Phone number",
            "medicine": "Medicine",
            "branch": "Branch",
            "contact_no": "Contact Number",
            "manager_contact": "Manager Contact",
            "store_name": "Store Name",
        }
        if alias_result.ambiguities:
            for canon, cols in alias_result.ambiguities.items():
                label = friendly_col_names.get(canon, "a required field")
                st.error(
                    f"Your file has more than one column for **{label}** "
                    f"({', '.join(cols)}). Please keep only one and re-upload."
                )
        if alias_result.missing_required:
            for req in alias_result.missing_required:
                label = friendly_col_names.get(req, req)
                st.error(f"Your file is missing a required column: **{label}**")
        st.error("Please update your file and upload it again.")
        st.stop()

    missing = check_image_campaign_columns(df)
    if missing:
        st.error(missing_image_columns_message(missing))
        st.stop()

    grouped_customers, invalid_df = validate_and_group_customers(df)

    total_rows = len(df)
    grouped_count = len(grouped_customers)
    invalid_count = len(invalid_df)

    store_name = os.getenv("MEDICAL_STORE_NAME", "PHARMA HUBB").strip() or "PHARMA HUBB"
    image_template_name = os.getenv("XINNO_IMAGE_TEMPLATE_NAME", "").strip()
    image_url = os.getenv("XINNO_IMAGE_URL", "").strip()
    template_language = os.getenv("WHATSAPP_TEMPLATE_LANGUAGE", "en").strip() or "en"

    # -------------------------------------------------------------------
    # Readiness summary (client-friendly)
    # -------------------------------------------------------------------
    st.markdown(f"#### {grouped_count + invalid_count} Customers")
    m1, m2 = st.columns(2)
    m1.metric("Ready", grouped_count)
    m2.metric("Need Attention", invalid_count)

    if grouped_count == 0 and invalid_count == 0:
        st.warning("No customer records found in the uploaded file.")
    elif grouped_count == 0:
        st.warning("No customers are ready to receive messages yet. Please review the records below.")

    if grouped_customers:
        st.dataframe(
            _client_customer_table(grouped_customers),
            use_container_width=True,
            hide_index=True,
        )

    if not invalid_df.empty:
        with st.expander("Customers that need attention", expanded=False):
            st.dataframe(
                _client_attention_table(invalid_df),
                use_container_width=True,
                hide_index=True,
            )

    if grouped_customers:
        with st.expander("Medicine list per customer", expanded=False):
            for cust in grouped_customers:
                phone_display = cust.get("Original Phone") or mask_phone_for_audit(
                    cust["Normalized Phone"]
                )
                st.markdown(
                    f"**{cust['Name']}** · {phone_display} · {cust.get('Branch', '')}"
                )
                if cust.get("Medicine List"):
                    st.write(cust["Medicine List"])
                else:
                    st.caption("No medicines listed")
                st.divider()

    # -------------------------------------------------------------------
    # Campaign Image
    # -------------------------------------------------------------------
    st.subheader("Campaign Image")
    st.caption("Choose the image that will appear at the top of each WhatsApp reminder.")

    image_source = st.radio(
        "How would you like to add the image?",
        options=["Use saved image", "Upload from device"],
        key="img_source_choice",
        horizontal=True,
    )

    selected_image_url: Optional[str] = None

    if image_source == "Upload from device":
        st.caption("Accepted formats: JPG, JPEG, PNG, WEBP (Max 10 MB)")
        uploaded_img = st.file_uploader(
            "Select an image",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_device_uploader",
        )
        if uploaded_img is not None:
            is_valid_img, val_msg = validate_image_file(uploaded_img, filename=uploaded_img.name)
            if not is_valid_img:
                st.error(f"This image cannot be used: {val_msg}")
            else:
                img_size_kb = len(uploaded_img.getvalue()) // 1024
                st.write(f"**Selected:** {uploaded_img.name} ({img_size_kb} KB)")
                st.image(uploaded_img, caption="Image preview", width=320)

                upload_clicked = st.button(
                    "Upload Image",
                    key="img_upload_to_cloudinary_btn",
                    type="secondary",
                )
                if upload_clicked:
                    with st.spinner("Preparing your campaign image..."):
                        upload_res = upload_image_to_cloudinary(
                            uploaded_img, filename=uploaded_img.name, reload_dotenv=True
                        )
                        if upload_res["success"]:
                            st.session_state.img_uploaded_cloudinary_url = upload_res["secure_url"]
                            st.session_state.img_uploaded_filename = uploaded_img.name
                            st.success("Image ready for this campaign.")
                        else:
                            st.error(
                                "We could not prepare this image. Please try again "
                                "or use a different image."
                            )

                if st.session_state.get("img_uploaded_cloudinary_url"):
                    selected_image_url = st.session_state.img_uploaded_cloudinary_url
                    st.success("Image ready")
                    st.image(selected_image_url, caption="Campaign image", width=320)
                else:
                    st.info("Click **Upload Image** to prepare this image for sending.")
        else:
            st.info("Select an image file to continue.")

    else:
        default_url = image_url or (
            "https://res.cloudinary.com/troli5kq/image/upload/f_auto,q_auto/"
            "PHARMA_HUBB_-_medicine_Refill_Reminder_Image_Template"
        )
        image_url_input = st.text_input(
            "Image link",
            value=default_url,
            key="img_image_url_input",
            help="Paste a public image link, or keep the default campaign image.",
        )
        cleaned_url = image_url_input.strip()
        is_valid_url, url_err = validate_image_url(cleaned_url)
        if is_valid_url:
            selected_image_url = cleaned_url
            try:
                st.image(selected_image_url, caption="Campaign image", width=320)
            except Exception:
                st.warning("The image link looks valid, but the preview could not be loaded.")
        else:
            st.error(f"Please provide a valid image link. ({url_err})")

    # -------------------------------------------------------------------
    # Review Message
    # -------------------------------------------------------------------
    st.subheader("Review Message")
    store_name = st.text_input("Pharmacy name", value=store_name, key="img_store_name")

    if grouped_count == 0:
        st.warning("Upload a valid customer list before reviewing or sending messages.")
    elif not selected_image_url:
        st.info("Add a campaign image above to see the full message preview.")
    else:
        samples = build_image_sample_previews(grouped_customers, store_name, limit=1)
        sample = samples[0]
        st.markdown("##### WhatsApp preview")
        st.caption(
            f"Preview for **{sample['customer_name']}** "
            f"({sample.get('original_phone') or mask_phone_for_audit(sample['normalized_phone'])}). "
            "Each customer will receive their own name, branch, and medicines."
        )
        st.image(selected_image_url, width=280)
        st.text_area(
            "Message",
            value=sample["message_preview"],
            height=300,
            disabled=True,
            key="img_whatsapp_message_preview",
            label_visibility="collapsed",
        )

        st.markdown("##### Send preview")
        st.metric("Eligible recipients", grouped_count)
        st.write(f"**Pharmacy:** {store_name}")
        st.write(f"**Selected Image URL:** {selected_image_url}")
        st.write(
            f"Total rows: {total_rows} · "
            f"Eligible (grouped): {grouped_count} · "
            f"Invalid: {invalid_count}"
        )

    # -------------------------------------------------------------------
    # Send WhatsApp Messages
    # -------------------------------------------------------------------
    st.subheader("Send WhatsApp Messages")

    if not image_template_name:
        st.error(
            "This campaign is not fully configured yet. "
            "Please contact your administrator before sending."
        )
    elif grouped_count == 0:
        st.warning("No customers are ready to receive messages.")
    elif not selected_image_url:
        st.info("Please add a campaign image before sending.")
    else:
        st.write(
            f"You are ready to send refill reminders to **{grouped_count}** customer"
            f"{'s' if grouped_count != 1 else ''}."
        )

        can_send = (
            grouped_count > 0
            and bool(selected_image_url)
            and bool(image_template_name)
            and not st.session_state.img_bulk_send_in_progress
            and not st.session_state.img_campaign_complete
        )

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
            open_confirm = st.button(
                "Send WhatsApp Messages",
                disabled=not can_send or st.session_state.img_show_confirm,
                key="img_bulk_send_btn",
                type="primary",
                use_container_width=True,
            )
            if open_confirm:
                st.session_state.img_show_confirm = True

        if st.session_state.img_show_confirm and not st.session_state.img_campaign_complete:
            st.warning(
                f"**Confirm Campaign**\n\n"
                f"You are about to send WhatsApp refill reminders to "
                f"**{grouped_count}** customer{'s' if grouped_count != 1 else ''}.\n\n"
                "These are real WhatsApp messages."
            )
            cancel_col, confirm_col = st.columns(2)
            with cancel_col:
                if st.button("Cancel", key="img_confirm_cancel", use_container_width=True):
                    st.session_state.img_show_confirm = False
                    st.rerun()
            with confirm_col:
                send_confirmed = st.button(
                    "Send Messages",
                    key="img_confirm_send",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.img_bulk_send_in_progress,
                )

            if send_confirmed and bulk_confirmation_ready(True):
                if not st.session_state.img_bulk_active_attempt_id:
                    st.session_state.img_bulk_active_attempt_id = new_bulk_attempt_id()
                bulk_id = st.session_state.img_bulk_active_attempt_id

                if (
                    bulk_id in st.session_state.img_bulk_completed_attempt_ids
                    or bulk_id in st.session_state.img_bulk_started_attempt_ids
                ):
                    st.error(
                        "This campaign was already started. "
                        "Click **Start New Campaign** to begin again."
                    )
                elif not selected_image_url:
                    st.error("A campaign image is required before sending.")
                elif not image_template_name:
                    st.error(
                        "This campaign is not fully configured yet. "
                        "Please contact your administrator."
                    )
                else:
                    st.session_state.img_bulk_started_attempt_ids.add(bulk_id)
                    st.session_state.img_bulk_send_in_progress = True
                    try:
                        progress = st.progress(
                            0.0, text="Sending WhatsApp Messages..."
                        )
                        status_box = st.empty()
                        status_box.info(
                            f"Sending WhatsApp Messages...\n\n"
                            f"0 of {grouped_count} messages processed"
                        )

                        customer_lookup = {
                            (c["Name"].strip().lower(), c["Normalized Phone"]): c
                            for c in grouped_customers
                        }

                        def _send_one(phone_number, customer_name, store_name, dry_run=False):
                            lookup_key = (
                                str(customer_name).strip().lower(),
                                str(phone_number).strip(),
                            )
                            cust = customer_lookup.get(lookup_key, {})
                            return send_image_template_message(
                                phone_number=phone_number,
                                customer_name=customer_name,
                                store_name=store_name,
                                branch=cust.get("Branch", ""),
                                medicine_list=cust.get("Medicine List", ""),
                                contact_no=cust.get("Contact No.", ""),
                                manager_contact=cust.get("Manager Contact", ""),
                                image_url=selected_image_url,
                                dry_run=False,
                            )

                        summary_result = execute_bulk_send(
                            grouped_customers,
                            store_name=store_name,
                            template_name=image_template_name,
                            template_language=template_language,
                            send_fn=_send_one,
                            dry_run=False,
                            bulk_attempt_id=bulk_id,
                        )

                        for rec in summary_result["records"]:
                            aid = rec["attempt_id"]
                            if aid not in st.session_state.img_recorded_attempt_ids:
                                st.session_state.img_send_history = append_send_history(
                                    st.session_state.img_send_history, rec
                                )
                                st.session_state.img_recorded_attempt_ids.add(aid)
                                LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                                log_audit_record_safely(rec, str(LOG_FILE))

                        attempted = summary_result["attempted"]
                        successful = summary_result["successful"]
                        failed = summary_result["failed"]
                        progress.progress(
                            1.0,
                            text=(
                                f"{attempted} of {grouped_count} messages processed · "
                                f"{successful} sent successfully · {failed} failed"
                            ),
                        )
                        status_box.empty()

                        st.session_state.img_bulk_completed_attempt_ids.add(bulk_id)
                        st.session_state.img_bulk_last_summary = summary_result
                        st.session_state.img_bulk_active_attempt_id = None
                        st.session_state.img_show_confirm = False
                        st.session_state.img_campaign_complete = True
                    finally:
                        st.session_state.img_bulk_send_in_progress = False

    # -------------------------------------------------------------------
    # Campaign results
    # -------------------------------------------------------------------
    if st.session_state.img_bulk_last_summary:
        st.divider()
        s = st.session_state.img_bulk_last_summary
        st.subheader("Campaign Complete")
        st.success(
            f"**{s['successful']}** message{'s' if s['successful'] != 1 else ''} "
            f"sent successfully"
        )
        if s["failed"]:
            st.error(
                f"**{s['failed']}** message{'s' if s['failed'] != 1 else ''} failed"
            )

        results_df = _client_results_table(s["records"])
        if not results_df.empty:
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download results",
                data=history_to_csv_bytes(s["records"], mask_phones=True),
                file_name="whatsapp_refill_campaign_results.csv",
                mime="text/csv",
                key="img_download_bulk_results",
            )

        if st.button("Start New Campaign", key="img_start_new_campaign", type="secondary"):
            _reset_campaign_session()
            st.rerun()

    # -------------------------------------------------------------------
    # Session history
    # -------------------------------------------------------------------
    st.subheader("Session history")
    st.caption("Current session only. Phone numbers are masked in the table and CSV export.")
    history_df = history_to_dataframe(st.session_state.img_send_history, mask_phones=True)
    if history_df.empty:
        st.info("No send attempts recorded in this session yet.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv_bytes = history_to_csv_bytes(st.session_state.img_send_history, mask_phones=True)
        st.download_button(
            label="Download session history",
            data=csv_bytes,
            file_name="image_campaign_session_history.csv",
            mime="text/csv",
            key="img_download_send_history",
        )

else:
    st.info("Upload a CSV or XLSX file to get started.")
    st.subheader("Session history")
    st.caption("Current session only. Phone numbers are masked in the table and CSV export.")
    history_df = history_to_dataframe(st.session_state.img_send_history, mask_phones=True)
    if history_df.empty:
        st.info("No send attempts recorded in this session yet.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv_bytes = history_to_csv_bytes(st.session_state.img_send_history, mask_phones=True)
        st.download_button(
            label="Download session history",
            data=csv_bytes,
            file_name="image_campaign_session_history.csv",
            mime="text/csv",
            key="img_download_send_history_empty",
        )
