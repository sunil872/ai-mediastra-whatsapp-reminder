"""Med-Sync (Multi-Prescription Synchronization & Appointment-Based Refill Bundling).

Enterprise engine that clusters multiple active chronic prescriptions for each patient
due within a configurable synchronization window (default <= 7 days) into a single,
coordinated refill appointment bundle. This minimizes patient notification fatigue,
reduces delivery friction, lowers operational messaging costs, and maximizes medication adherence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Union
import uuid

import pandas as pd

from refillcare.engine.decision_types import RefillDecision

logger = logging.getLogger("MedSyncEngine")


@dataclass
class SyncedMedicationItem:
    """Individual medication within a patient's synchronized refill bundle."""

    item_id: str
    item_name: str
    last_purchase_date: date
    expected_refill_date: date
    predicted_interval_days: Optional[int]
    stability_tier: str
    dos_days: Optional[float] = None
    quantile_p10_date: Optional[date] = None
    quantile_p90_date: Optional[date] = None
    is_anchor: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.last_purchase_date:
            d["last_purchase_date"] = self.last_purchase_date.strftime("%Y-%m-%d")
        if self.expected_refill_date:
            d["expected_refill_date"] = self.expected_refill_date.strftime("%Y-%m-%d")
        if self.quantile_p10_date:
            d["quantile_p10_date"] = self.quantile_p10_date.strftime("%Y-%m-%d")
        if self.quantile_p90_date:
            d["quantile_p90_date"] = self.quantile_p90_date.strftime("%Y-%m-%d")
        return d


@dataclass
class MedSyncBundle:
    """Synchronized multi-prescription reminder bundle for a patient."""

    bundle_id: str
    customer_id: str
    customer_name: str
    mobile_no: Optional[str]
    anchor_item_id: str
    anchor_item_name: str
    anchor_refill_date: date
    window_start_date: date
    window_end_date: date
    synced_items: List[SyncedMedicationItem] = field(default_factory=list)
    total_items_count: int = 0
    message_reduction_count: int = 0
    bundled_message_text: str = ""
    earliest_p10_date: Optional[date] = None
    latest_p90_date: Optional[date] = None
    lifecycle_tier: str = "DUE"  # DUE, FOLLOWUP, or LAPSED
    refill_function: str = "DUE_REFILL"  # DUE_REFILL, REFILL_FOLLOW_UP, or LAPSED_REENGAGEMENT
    status: str = "PENDING"
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["anchor_refill_date"] = self.anchor_refill_date.strftime("%Y-%m-%d")
        d["window_start_date"] = self.window_start_date.strftime("%Y-%m-%d")
        d["window_end_date"] = self.window_end_date.strftime("%Y-%m-%d")
        d["lifecycle_tier"] = self.lifecycle_tier
        
        # Determine standardized refill function
        if self.lifecycle_tier == "FOLLOWUP":
            resolved_func = "REFILL_FOLLOW_UP"
        elif self.lifecycle_tier == "LAPSED":
            resolved_func = "LAPSED_REENGAGEMENT"
        else:
            resolved_func = "DUE_REFILL"
        d["refill_function"] = self.refill_function or resolved_func

        if self.earliest_p10_date:
            d["earliest_p10_date"] = self.earliest_p10_date.strftime("%Y-%m-%d")
        if self.latest_p90_date:
            d["latest_p90_date"] = self.latest_p90_date.strftime("%Y-%m-%d")
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        d["synced_items"] = [item.to_dict() for item in self.synced_items]
        return d


class MedSyncEngine:
    """Synchronizes multi-prescription refill reminders for pharmacy patients."""

    def __init__(self, sync_window_days: int = 0, default_pharmacy_name: str = "Mediastra Pharmacy"):
        self.sync_window_days = sync_window_days
        self.default_pharmacy_name = default_pharmacy_name

    def cluster_patient_decisions(
        self,
        decisions: Union[List[RefillDecision], pd.DataFrame],
        sync_window_days: Optional[int] = None,
        pharmacy_name: Optional[str] = None,
    ) -> List[MedSyncBundle]:
        """Group patient refill decisions into synchronized multi-medication bundles.

        Args:
            decisions: List of RefillDecision objects or DataFrame of decisions.
            sync_window_days: Maximum interval gap in days between items to bundle together (default 0).
            pharmacy_name: Pharmacy brand name for message templating.

        Returns:
            List of MedSyncBundle instances (including singletons and multi-item bundles).
        """
        window = sync_window_days if sync_window_days is not None else self.sync_window_days
        pharmacy = pharmacy_name or self.default_pharmacy_name

        # Standardize input to list of normalized records
        raw_records: List[Dict[str, Any]] = []
        if isinstance(decisions, pd.DataFrame):
            raw_records = decisions.to_dict(orient="records")
        elif isinstance(decisions, list):
            for d in decisions:
                if isinstance(d, RefillDecision):
                    raw_records.append(d.to_dict())
                elif isinstance(d, dict):
                    raw_records.append(d)
                else:
                    try:
                        raw_records.append(asdict(d))
                    except Exception:
                        pass

        def _extract_field(rec: Dict[str, Any], *keys: str, default: Any = "") -> Any:
            for k in keys:
                if k in rec and rec[k] is not None:
                    val = rec[k]
                    val_str = str(val).strip()
                    if val_str != "" and val_str.lower() not in ("nan", "none", "nat", "<na>"):
                        return val
            return default

        def _parse_date(val: Any) -> Optional[date]:
            if val is None:
                return None
            if isinstance(val, date) and not isinstance(val, datetime):
                return val
            if isinstance(val, (datetime, pd.Timestamp)):
                return val.date()
            if isinstance(val, str):
                s = val.strip()
                if not s or s.lower() in ("nan", "none", "-", "nat"):
                    return None
                try:
                    parsed = pd.to_datetime(s, errors="coerce")
                    if not pd.isna(parsed):
                        return parsed.date()
                except Exception:
                    pass
            return None

        # Normalize and filter active records with valid expected_refill_date
        valid_records: List[Dict[str, Any]] = []
        for r in raw_records:
            if not r.get("is_eligible", True):
                continue

            erd_raw = _extract_field(
                r,
                "expected_refill_date",
                "Expected Refill Date",
                "expected_refill_date_raw",
                "raw_reminder_date",
                "reminder_date",
            )
            erd = _parse_date(erd_raw)
            if not erd:
                continue

            cid = str(
                _extract_field(
                    r,
                    "customer_id",
                    "customerId",
                    "Customer ID",
                    "partycode",
                    "patient_id",
                    "Customer Name",
                    "customerName",
                    "Customer",
                    default=f"CUST_{uuid.uuid4().hex[:8]}",
                )
            ).strip()

            cname = str(
                _extract_field(
                    r,
                    "customer_name",
                    "customerName",
                    "Customer Name",
                    "Customer",
                    "partyname",
                    "patient_name",
                    default=cid,
                )
            ).strip()

            item_id = str(
                _extract_field(
                    r,
                    "item_id",
                    "itemId",
                    "Item ID",
                    "itemcode",
                    "drug_id",
                    default="MED_ITEM",
                )
            ).strip()

            item_name = str(
                _extract_field(
                    r,
                    "item_name",
                    "itemName",
                    "Item Name",
                    "Medication",
                    "Medicine",
                    "medication_name",
                    "itemdescription",
                    default=item_id,
                )
            ).strip()

            mobile = str(
                _extract_field(
                    r,
                    "mobile_no",
                    "MOBILE_NO",
                    "Mobile Number",
                    "Delivery Phone",
                    "phone_number",
                    "mobile",
                    default="",
                )
            ).strip()

            lpd_raw = _extract_field(
                r,
                "last_purchase_date",
                "Last Purchase Date",
                "latest_purchase_date",
                "invoice_date",
                "latest_purchase_date_raw",
            )
            lpd = _parse_date(lpd_raw) or (erd - timedelta(days=30))

            p10_raw = _extract_field(r, "quantile_p10_date", "p10_date")
            p90_raw = _extract_field(r, "quantile_p90_date", "p90_date")

            stability = str(
                _extract_field(
                    r,
                    "stability_tier",
                    "Stability Tier",
                    "Pilot Tier",
                    "history_quality",
                    default="MEDIUM-SAFE",
                )
            ).strip()

            dos_val = _extract_field(
                r,
                "dos_days",
                "estimated_days_of_supply",
                "Estimated Days of Supply",
                "predicted_days_until_refill",
                default=30.0,
            )
            try:
                dos_days = float(dos_val)
            except Exception:
                dos_days = 30.0

            pred_interval = _extract_field(
                r,
                "predicted_interval_days",
                "predicted_days_until_refill",
                "estimated_days_of_supply",
                "Estimated Days of Supply",
                default=int(dos_days),
            )
            try:
                pred_interval_days = int(round(float(pred_interval)))
            except Exception:
                pred_interval_days = 30

            stage_offset_val = _extract_field(r, "stage_offset", "Stage Offset", default=None)
            if stage_offset_val is not None:
                try:
                    st_int = int(stage_offset_val)
                    if st_int <= 0:
                        lifecycle_tier = "DUE"
                    elif st_int in (2, 5):
                        lifecycle_tier = "FOLLOWUP"
                    elif st_int in (40, 45):
                        lifecycle_tier = "LAPSED"
                    else:
                        lifecycle_tier = "DUE"
                except Exception:
                    lifecycle_tier = "DUE"
            else:
                lifecycle_tier = "DUE"

            clean_mobile = mobile if (mobile and str(mobile).strip().lower() not in ("nan", "none", "<na>", "null", "")) else "-"

            # Primary patient key: (Customer Name + Mobile Number)
            # This ensures distinct customers with identical names but different phone numbers
            # are preserved as separate patients, while '-' / blank numbers are also preserved.
            patient_key = f"{cname}::{clean_mobile}"

            norm_rec = {
                "customer_id": cid,
                "customer_name": cname,
                "mobile_no": clean_mobile if clean_mobile != "-" else "-",
                "patient_key": patient_key,
                "item_id": item_id,
                "item_name": item_name,
                "last_purchase_date": lpd,
                "expected_refill_date": erd,
                "predicted_interval_days": pred_interval_days,
                "stability_tier": stability,
                "dos_days": dos_days,
                "quantile_p10_date": _parse_date(p10_raw),
                "quantile_p90_date": _parse_date(p90_raw),
                "lifecycle_tier": lifecycle_tier,
                "_parsed_erd": erd,
            }
            valid_records.append(norm_rec)

        if not valid_records:
            return []

        # Group by (patient_key, lifecycle_tier) to guarantee patient & stage isolation
        customer_tier_groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in valid_records:
            key = (r["patient_key"], r["lifecycle_tier"])
            customer_tier_groups.setdefault(key, []).append(r)

        bundles: List[MedSyncBundle] = []

        for (pkey, tier), cust_items in customer_tier_groups.items():
            # Sort customer items chronologically by expected_refill_date
            cust_items.sort(key=lambda x: x["_parsed_erd"])

            # Resolve best known customer_id, customer name and mobile across items
            cid = next((it["customer_id"] for it in cust_items if it["customer_id"]), pkey.split("::")[0])
            cust_name = next((it["customer_name"] for it in cust_items if it["customer_name"] and it["customer_name"] != cid), pkey.split("::")[0])
            best_mobile = next((it["mobile_no"] for it in cust_items if it["mobile_no"] and it["mobile_no"] != "-"), "-")

            # Cluster items using a greedy temporal window
            clusters: List[List[Dict[str, Any]]] = []
            current_cluster: List[Dict[str, Any]] = []

            for item in cust_items:
                if not current_cluster:
                    current_cluster.append(item)
                else:
                    cluster_anchor_date = current_cluster[0]["_parsed_erd"]
                    item_date = item["_parsed_erd"]
                    if (item_date - cluster_anchor_date).days <= window:
                        current_cluster.append(item)
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [item]

            if current_cluster:
                clusters.append(current_cluster)

            # Build MedSyncBundle for each cluster
            for cluster_idx, cluster in enumerate(clusters):
                # Anchor is typically the earliest or highest-stability item
                def _sort_key(x: Dict[str, Any]):
                    tier_priority = {"HIGH": 0, "MEDIUM-SAFE": 1, "MEDIUM-RISK": 2, "UNSTABLE": 3}
                    tier_rank = tier_priority.get(x.get("stability_tier", ""), 4)
                    return (tier_rank, x["_parsed_erd"])

                sorted_by_priority = sorted(cluster, key=_sort_key)
                anchor = sorted_by_priority[0]
                anchor_date: date = anchor["_parsed_erd"]

                synced_items_list: List[SyncedMedicationItem] = []
                p10_dates: List[date] = []
                p90_dates: List[date] = []

                for itm in cluster:
                    is_anchor = (itm["item_id"] == anchor["item_id"])
                    p10_d = itm.get("quantile_p10_date")
                    p90_d = itm.get("quantile_p90_date")
                    if p10_d:
                        p10_dates.append(p10_d)
                    if p90_d:
                        p90_dates.append(p90_d)

                    synced_item = SyncedMedicationItem(
                        item_id=itm["item_id"],
                        item_name=itm["item_name"],
                        last_purchase_date=itm["last_purchase_date"],
                        expected_refill_date=itm["_parsed_erd"],
                        predicted_interval_days=itm["predicted_interval_days"],
                        stability_tier=itm["stability_tier"],
                        dos_days=itm["dos_days"],
                        quantile_p10_date=p10_d,
                        quantile_p90_date=p90_d,
                        is_anchor=is_anchor,
                    )
                    synced_items_list.append(synced_item)

                total_items = len(synced_items_list)
                reduction = max(0, total_items - 1)
                window_start = cluster[0]["_parsed_erd"]
                window_end = cluster[-1]["_parsed_erd"]

                earliest_p10 = min(p10_dates) if p10_dates else None
                latest_p90 = max(p90_dates) if p90_dates else None

                tier_suffix = f"_{tier.lower()}" if tier != "DUE" else ""
                clean_mobile_tag = best_mobile.replace("+", "").replace("-", "BLANK") if best_mobile and best_mobile != "-" else "BLANK"
                bundle_uid = f"SYNC_{cid}_{clean_mobile_tag}_{anchor_date.strftime('%Y%m%d')}{tier_suffix}_{cluster_idx+1}"

                if tier == "FOLLOWUP":
                    func_tag = "REFILL_FOLLOW_UP"
                elif tier == "LAPSED":
                    func_tag = "LAPSED_REENGAGEMENT"
                else:
                    func_tag = "DUE_REFILL"

                bundle = MedSyncBundle(
                    bundle_id=bundle_uid,
                    customer_id=cid,
                    customer_name=cust_name,
                    mobile_no=best_mobile,
                    anchor_item_id=anchor["item_id"],
                    anchor_item_name=anchor["item_name"],
                    anchor_refill_date=anchor_date,
                    window_start_date=window_start,
                    window_end_date=window_end,
                    synced_items=synced_items_list,
                    total_items_count=total_items,
                    message_reduction_count=reduction,
                    earliest_p10_date=earliest_p10,
                    latest_p90_date=latest_p90,
                    lifecycle_tier=tier,
                    refill_function=func_tag,
                )
                bundle.bundled_message_text = self.generate_bundle_whatsapp_message(bundle, pharmacy_name=pharmacy)
                bundles.append(bundle)

        logger.info(
            f"Med-Sync generated {len(bundles)} bundles for {len(customer_tier_groups)} patient-stage groups "
            f"(Saved {sum(b.message_reduction_count for b in bundles)} individual messages)."
        )
        return bundles

    def generate_bundle_whatsapp_message(
        self,
        bundle: MedSyncBundle,
        pharmacy_name: str = "Mediastra Pharmacy",
        hotline: str = "919876543210",
    ) -> str:
        """Generate clinical WhatsApp synchronized refill reminder message based on lifecycle tier."""
        anchor_date_str = bundle.anchor_refill_date.strftime("%d %b %Y")
        tier = getattr(bundle, "lifecycle_tier", "DUE")

        # 1. LAPSED RE-ENGAGEMENT TIER (+45 Days)
        if tier == "LAPSED":
            if bundle.total_items_count == 1:
                item = bundle.synced_items[0]
                return (
                    f"Hello {bundle.customer_name},\n\n"
                    f"This is a care check-in from *{pharmacy_name}*.\n"
                    f"We noticed your regular medication *{item.item_name}* was due for refill around *{anchor_date_str}*. "
                    f"Have you refilled elsewhere, or would you like us to assist with home delivery?\n\n"
                    f"Reply *YES* to place an order, or call us directly at {hotline}."
                )
            items_bullets = "\n".join([f"  • {item.item_name}" for item in bundle.synced_items])
            return (
                f"Hello {bundle.customer_name},\n\n"
                f"🏥 *{pharmacy_name} - Chronic Care Check-in Notice*\n\n"
                f"We noticed your regular prescription refills due around *{anchor_date_str}* have not been refilled yet:\n\n"
                f"{items_bullets}\n\n"
                f"📦 Would you like our clinical team to assist you with home delivery or prescription coordination?\n\n"
                f"• Reply *1* for Refill & Delivery\n"
                f"• Reply *2* if already refilled or paused\n"
                f"• Call our pharmacist hotline: {hotline}"
            )

        # 2. FOLLOW-UP TIER (+2, +5 Days)
        if tier == "FOLLOWUP":
            if bundle.total_items_count == 1:
                item = bundle.synced_items[0]
                return (
                    f"Hello {bundle.customer_name},\n\n"
                    f"This is a friendly follow-up from *{pharmacy_name}*.\n"
                    f"Your medication *{item.item_name}* was due for refill on *{anchor_date_str}*.\n\n"
                    f"Reply *YES* to confirm your refill for pickup or delivery, or call us directly at {hotline}."
                )
            items_bullets = "\n".join([f"  • {item.item_name}" for item in bundle.synced_items])
            return (
                f"Hello {bundle.customer_name},\n\n"
                f"🏥 *{pharmacy_name} - Follow-up Refill Reminder*\n\n"
                f"Friendly reminder regarding your ongoing chronic refills due on *{anchor_date_str}*:\n\n"
                f"{items_bullets}\n\n"
                f"📦 We have reserved fresh stock for your medications.\n\n"
                f"• Reply *1* to Confirm All {bundle.total_items_count} items\n"
                f"• Reply *2* to Customize or Delay\n"
                f"• Call our pharmacist hotline: {hotline}"
            )

        # 3. DUE / ADVANCE TIER (Default: -7, -3, -1, 0 Days)
        if bundle.total_items_count == 1:
            item = bundle.synced_items[0]
            return (
                f"Hello {bundle.customer_name},\n\n"
                f"This is a gentle refill reminder from *{pharmacy_name}*.\n"
                f"Your medication *{item.item_name}* is due for refill around *{anchor_date_str}*.\n\n"
                f"Reply *YES* to place your refill order for pickup or delivery, or call us directly at {hotline}."
            )

        items_bullets = "\n".join([f"  • {item.item_name}" for item in bundle.synced_items])
        return (
            f"Hello {bundle.customer_name},\n\n"
            f"🏥 *{pharmacy_name} - Synchronized Prescription Refill Notice*\n\n"
            f"To save you multiple pharmacy trips, we have synchronized your upcoming chronic refills due around *{anchor_date_str}* into a single convenient order:\n\n"
            f"{items_bullets}\n\n"
            f"📦 Would you like us to prepare all {bundle.total_items_count} medications together?\n\n"
            f"• Reply *1* to Confirm All {bundle.total_items_count} items\n"
            f"• Reply *2* to Customize or Delay\n"
            f"• Call our pharmacist hotline: {hotline}"
        )

    def summarize_sync_impact(self, bundles: List[MedSyncBundle]) -> Dict[str, Any]:
        """Calculate operational and economic impact metrics of Med-Sync bundling."""
        total_bundles = len(bundles)
        multi_item_bundles = [b for b in bundles if b.total_items_count > 1]
        single_item_bundles = [b for b in bundles if b.total_items_count == 1]
        
        total_prescriptions = sum(b.total_items_count for b in bundles)
        total_messages_saved = sum(b.message_reduction_count for b in bundles)
        
        friction_reduction_pct = (
            round((total_messages_saved / total_prescriptions) * 100, 2)
            if total_prescriptions > 0
            else 0.0
        )
        
        max_items_in_bundle = max((b.total_items_count for b in bundles), default=0)
        avg_items_per_bundle = round(total_prescriptions / max(1, total_bundles), 2)

        return {
            "total_patients_analyzed": len(set(b.customer_id for b in bundles)),
            "total_prescriptions_synced": total_prescriptions,
            "total_dispatches_generated": total_bundles,
            "multi_item_bundles_count": len(multi_item_bundles),
            "single_item_bundles_count": len(single_item_bundles),
            "multi_item_bundle_rate_pct": round(len(multi_item_bundles) / max(1, total_bundles) * 100, 2),
            "individual_messages_saved": total_messages_saved,
            "message_reduction_rate_pct": friction_reduction_pct,
            "max_items_in_single_bundle": max_items_in_bundle,
            "avg_items_per_bundle": avg_items_per_bundle,
        }
