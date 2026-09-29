"""Detailed August 2026 Actual vs Predicted Gap and Accuracy Analysis.

Answers:
1. Exact count of correct vs wrong under different tolerance thresholds (+-0d, +-1d, +-3d, +-7d, +-14d, +-30d).
2. When are predictions BEFORE actual (early prediction / patient arrived later) vs AFTER actual (late prediction / patient arrived earlier).
3. Mean and median gap between predicted date and actual date.
4. Real-world human behavioral context (leftover supply, gap in-between, buying outside, early top-up, dose change).
5. Concrete sample customer cases from August 2026 illustrating each scenario.
"""

from __future__ import annotations
import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refillcare.data.packing import parse_pack_units
from refillcare.features.point_in_time import POINT_IN_TIME_FEATURE_COLS
from refillcare.models.supply_hybrid import load_human_ml_bundle

DATA_PATH = Path("data/refillcare/processed/clean_transactions.parquet")


def run_detailed_analysis():
    print(f"Loading data from {DATA_PATH}...")
    t0 = time.time()
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    clean = df[(df["refillcare_eligible"] == True) & (df["customerId"].notna()) & (df["itemId"].notna())].copy()
    clean["invoice_date"] = pd.to_datetime(clean["invoice_date"])

    # Parse pack units
    clean["pack_units"] = clean["packing"].apply(lambda p: float(parse_pack_units(p) or 10.0))
    clean["total_units"] = clean["quantity"] * clean["pack_units"]

    # Deduplicate / aggregate same-day transactions per customer-item
    agg_df = (
        clean.groupby(["customerId", "itemId", "invoice_date"])
        .agg(
            total_units=("total_units", "sum"),
            pack_units=("pack_units", "first"),
            customerName=("customerName", "first"),
            itemName=("itemName", "first"),
        )
        .reset_index()
        .sort_values(["customerId", "itemId", "invoice_date"])
    )

    # Vectorized point-in-time lags
    agg_df["prev_date"] = agg_df.groupby(["customerId", "itemId"])["invoice_date"].shift(1)
    agg_df["prev_units"] = agg_df.groupby(["customerId", "itemId"])["total_units"].shift(1)
    agg_df["prev_prev_date"] = agg_df.groupby(["customerId", "itemId"])["invoice_date"].shift(2)
    agg_df["prev_prev_units"] = agg_df.groupby(["customerId", "itemId"])["total_units"].shift(2)
    agg_df["next_date"] = agg_df.groupby(["customerId", "itemId"])["invoice_date"].shift(-1)

    # Intervals
    agg_df["target_interval"] = (agg_df["next_date"] - agg_df["invoice_date"]).dt.days
    agg_df["last_interval"] = (agg_df["invoice_date"] - agg_df["prev_date"]).dt.days
    agg_df["prior_interval"] = (agg_df["prev_date"] - agg_df["prev_prev_date"]).dt.days

    # Filter strictly to August 2026 holdout test set
    test_mask = (
        (agg_df["next_date"] >= "2026-08-01")
        & (agg_df["next_date"] <= "2026-08-31")
        & (agg_df["invoice_date"] <= "2026-07-31")
        & (agg_df["target_interval"] >= 5)
        & (agg_df["target_interval"] <= 180)
        & agg_df["prev_date"].notna()
        & (agg_df["last_interval"] > 0)
    )

    test_df = agg_df[test_mask].copy()

    # Historical stats strictly prior to 2026-08-01
    hist_stats = (
        agg_df[agg_df["invoice_date"] <= "2026-07-31"]
        .groupby(["customerId", "itemId"])
        .agg(
            hist_median=("last_interval", "median"),
            typical_units=("total_units", "median"),
            total_purchases=("invoice_date", "count"),
        )
        .reset_index()
    )

    test_df = test_df.merge(hist_stats, on=["customerId", "itemId"], how="left")
    test_df["hist_median"] = test_df["hist_median"].fillna(30.0).clip(7, 120)
    test_df["typical_units"] = test_df["typical_units"].fillna(30.0).clip(5, 180)

    # Filter to established customers (purchase count >= 4)
    test_df = test_df[test_df["total_purchases"] >= 4].copy().reset_index(drop=True)
    N = len(test_df)
    print(f"Established Path A Customer Cohort (August 2026): {N:,} actual transitions")

    # Strategy A: Baseline Historical Median
    test_df["pred_median"] = test_df["hist_median"].round()

    # Strategy D: Generalized Human-Level Hybrid Engine
    test_df["latest_units"] = test_df["total_units"].clip(5, 300)
    test_df["previous_units"] = test_df["prev_units"].fillna(test_df["latest_units"]).clip(5, 300)
    test_df["typical_units_median"] = test_df["typical_units"]
    test_df["recent_units_median"] = ((test_df["latest_units"] + test_df["previous_units"]) / 2.0).clip(5, 300)
    test_df["quantity_ratio_vs_typical"] = (test_df["latest_units"] / test_df["typical_units_median"]).clip(0.1, 10.0)
    test_df["quantity_ratio_vs_previous"] = (test_df["latest_units"] / test_df["previous_units"]).clip(0.1, 10.0)
    test_df["quantity_std"] = (test_df["latest_units"] - test_df["previous_units"]).abs()

    test_df["last_interval"] = test_df["last_interval"].clip(5, 180)
    test_df["prior_interval"] = test_df["prior_interval"].fillna(test_df["last_interval"]).clip(5, 180)
    test_df["historical_median_interval"] = test_df["hist_median"]
    test_df["recent_median_interval"] = test_df["last_interval"]
    test_df["rolling_interval_mean"] = test_df["hist_median"]
    test_df["cadence_norm_mad"] = ((test_df["last_interval"] - test_df["hist_median"]).abs() / test_df["hist_median"]).clip(0.0, 2.0)
    test_df["cadence_drift"] = (test_df["last_interval"] - test_df["hist_median"]).abs()

    test_df["estimated_consumption_velocity"] = (test_df["previous_units"] / test_df["last_interval"]).clip(0.33, 3.0)
    test_df["estimated_dos"] = (test_df["latest_units"] / test_df["estimated_consumption_velocity"]).clip(5.0, 365.0)

    consumed = test_df["last_interval"] * test_df["estimated_consumption_velocity"]
    residual = np.maximum(0.0, test_df["previous_units"] - consumed)
    test_df["estimated_residual_inventory"] = np.where(test_df["last_interval"] <= 20.0, residual, 0.0).clip(0, 90)
    test_df["estimated_effective_supply"] = test_df["latest_units"] + test_df["estimated_residual_inventory"]
    test_df["estimated_supply_days"] = (test_df["estimated_effective_supply"] / test_df["estimated_consumption_velocity"]).clip(5.0, 365.0)

    test_df["is_stocking_up"] = (test_df["latest_units"] >= 50.0).astype(float)
    test_df["is_partial_purchase"] = (test_df["latest_units"] <= 15.0).astype(float)
    test_df["is_early_topup"] = (test_df["last_interval"] <= 20.0).astype(float)
    test_df["is_post_lapse"] = (test_df["last_interval"] >= 45.0).astype(float)
    test_df["is_quantity_anomaly"] = ((test_df["quantity_ratio_vs_previous"] >= 2.5) | (test_df["quantity_ratio_vs_previous"] <= 0.35)).astype(float)
    test_df["purchase_count"] = test_df["total_purchases"].astype(float)
    test_df["customer_tenure_days"] = 180.0
    test_df["purchase_month"] = test_df["invoice_date"].dt.month.astype(float)

    human_bundle = load_human_ml_bundle()
    if human_bundle:
        raw_ml = human_bundle["model"].predict(test_df[POINT_IN_TIME_FEATURE_COLS])
    else:
        raw_ml = test_df["estimated_supply_days"].values

    min_b = np.maximum(7.0, test_df["estimated_supply_days"].values * 0.65)
    max_b = np.maximum(10.0, test_df["estimated_supply_days"].values * 1.50)
    clamped_ml = np.asarray(np.clip(raw_ml, min_b, max_b))

    hybrid_preds = []
    reasons = []
    for i in range(N):
        c_pred = clamped_ml[i]
        qr = test_df.at[i, "quantity_ratio_vs_typical"]
        hm = test_df.at[i, "hist_median"]
        lu = test_df.at[i, "latest_units"]
        ie = test_df.at[i, "is_early_topup"]
        sd = test_df.at[i, "estimated_supply_days"]
        delta = abs(raw_ml[i] - sd)

        if qr < 0.8:
            sc = round(hm * qr)
            p = max(5.0, min(float(sc), float(lu)))
            r = "Partial Purchase"
        elif ie == 1.0:
            sc = round(sd * 0.8)
            p = max(c_pred, float(sc), 45.0 if lu >= 50.0 else 20.0)
            r = "Early Top-Up Carryover"
        elif qr > 1.3 and lu < 90.0:
            sc = round(hm * qr)
            p = max(c_pred, float(sc), float(hm))
            r = "Multi-Pack Scaled"
        elif test_df.at[i, "is_post_lapse"] == 1.0:
            p = c_pred
            r = "Post-Lapse Reset"
        elif delta > 45.0 and lu >= 60.0:
            p = c_pred
            r = "Severe Divergence"
        else:
            p = c_pred
            r = "Consensus"
        hybrid_preds.append(round(p))
        reasons.append(r)

    test_df["pred_hybrid"] = hybrid_preds
    test_df["reason"] = reasons

    # Error analysis
    y_actual = test_df["target_interval"].values
    y_pred_h = test_df["pred_hybrid"].values
    y_pred_m = test_df["pred_median"].values

    # Gap definition: (Predicted Interval - Actual Interval)
    # If gap > 0: Predicted date was AFTER actual visit (patient came EARLIER than predicted)
    # If gap < 0: Predicted date was BEFORE actual visit (patient came LATER than predicted)
    gap_h = y_pred_h - y_actual
    abs_gap_h = np.abs(gap_h)

    test_df["gap_hybrid"] = gap_h
    test_df["abs_gap_hybrid"] = abs_gap_h

    print("\n" + "=" * 80)
    print("DETAILED RESULTS: ACTUAL vs PREDICTED IN AUGUST 2026")
    print("=" * 80)

    print(f"\n[1] OVERALL SUMMARY (Cohort: {N:,} transitions):")
    print(f"  - Mean Absolute Gap (MAE):     {np.mean(abs_gap_h):.2f} days")
    print(f"  - Median Absolute Gap:         {np.median(abs_gap_h):.1f} days")
    print(f"  - Standard Deviation of Gap:   {np.std(abs_gap_h):.2f} days")

    print(f"\n[2] ACCURACY & CORRECTNESS BREAKDOWN (How many are correct vs wrong?):")
    thresholds = [
        ("Exact Same Day (0 days gap)", abs_gap_h == 0),
        ("Within +-1 Day (Extremely Tight)", abs_gap_h <= 1),
        ("Within +-3 Days (Rough Estimation / Correct)", abs_gap_h <= 3),
        ("Within +-5 Days (Standard Grace Period)", abs_gap_h <= 5),
        ("Within +-7 Days (Clinical Adherence Window)", abs_gap_h <= 7),
        ("Within +-10 Days", abs_gap_h <= 10),
        ("Within +-14 Days (Bi-Weekly Margin)", abs_gap_h <= 14),
        ("Within +-21 Days (3-Week Margin)", abs_gap_h <= 21),
        ("Within +-30 Days (Monthly Margin)", abs_gap_h <= 30),
        ("Wrong / Large Gap (> 30 Days)", abs_gap_h > 30),
    ]

    for label, mask in thresholds:
        cnt = int(mask.sum())
        pct = (cnt / N) * 100
        print(f"  * {label:<45}: {cnt:>5,} | {pct:>5.1f}%")

    print(f"\n[3] TIMING DIRECTION (Was medication taken Before, On, or After Predicted Date?):")
    ontime_mask = abs_gap_h <= 3
    later_mask = gap_h < -3   # Predicted date was BEFORE actual purchase (patient came LATER)
    earlier_mask = gap_h > 3  # Predicted date was AFTER actual purchase (patient came EARLIER)

    n_ontime = int(ontime_mask.sum())
    n_later = int(later_mask.sum())
    n_earlier = int(earlier_mask.sum())

    print(f"\n  A. ON-TIME (Roughly Same Day: within +-3 days):")
    print(f"     Count: {n_ontime:,} patients ({n_ontime/N*100:.1f}%)")
    print(f"     Status: HIGH ACCURACY. Patient visited the pharmacy right on schedule.")

    print(f"\n  B. PREDICTED DATE BEFORE ACTUAL VISIT (Patient came LATER than expected):")
    print(f"     Count: {n_later:,} patients ({n_later/N*100:.1f}%)")
    gaps_l = -gap_h[later_mask]
    print(f"     Mean Gap: {np.mean(gaps_l):.1f} days late | Median Gap: {np.median(gaps_l):.1f} days late")
    print(f"     Clinical & Behavioral Causes:")
    print(f"       1. Gap in-between / Non-adherence: Missed taking doses, took medicine alternate days.")
    print(f"       2. Bought outside: Purchased 1-2 emergency strips from another medical store/chemist.")
    print(f"       3. Had leftover stock: Took top-up from spouse/family member or previous extra strip.")
    print(f"       4. Travel / Hospitalization: Delayed visit due to travel, festivals, or clinic checkups.")

    print(f"\n  C. PREDICTED DATE AFTER ACTUAL VISIT (Patient came EARLIER than expected):")
    print(f"     Count: {n_earlier:,} patients ({n_earlier/N*100:.1f}%)")
    gaps_e = gap_h[earlier_mask]
    print(f"     Mean Gap: {np.mean(gaps_e):.1f} days early | Median Gap: {np.median(gaps_e):.1f} days early")
    print(f"     Clinical & Behavioral Causes:")
    print(f"       1. Early Top-Up / Stocking Up: Patient came early before supply ran out (pre-travel).")
    print(f"       2. Dosage Escalation: Doctor increased regimen (e.g. from once-daily to twice-daily).")
    print(f"       3. Accompanying Visit: Came to buy other medicines and topped up chronic meds simultaneously.")
    print(f"       4. Shared medication: Gave portion of strips to household family member.")

    print(f"\n[4] DISTRIBUTION OF ERROR GAPS (Actual Interval vs Predicted Interval):")
    gap_ranges = [
        ("0 - 3 days", abs_gap_h <= 3),
        ("4 - 7 days", (abs_gap_h >= 4) & (abs_gap_h <= 7)),
        ("8 - 14 days", (abs_gap_h >= 8) & (abs_gap_h <= 14)),
        ("15 - 30 days", (abs_gap_h >= 15) & (abs_gap_h <= 30)),
        ("31 - 60 days", (abs_gap_h >= 31) & (abs_gap_h <= 60)),
        ("> 60 days (Severe Discontinuation/Lapse)", abs_gap_h > 60),
    ]
    for r_label, r_mask in gap_ranges:
        r_cnt = int(r_mask.sum())
        print(f"  * Gap {r_label:<42}: {r_cnt:>5,} ({r_cnt/N*100:>5.1f}%)")

    print("\n[5] REAL SAMPLES FROM AUGUST 2026 ACTUAL DATA:")
    print("-" * 115)
    print(f"{'Customer ID':<10} | {'Customer Name':<18} | {'Medicine Name':<24} | {'Actual':<6} | {'Pred':<5} | {'Gap':<6} | {'Scenario'}")
    print("-" * 115)

    # 1. Exact
    s_exact = test_df[test_df["abs_gap_hybrid"] <= 1].head(3)
    for _, r in s_exact.iterrows():
        print(f"{r['customerId']:<10} | {str(r['customerName'])[:18]:<18} | {str(r['itemName'])[:24]:<24} | {int(r['target_interval']):>4}d | {int(r['pred_hybrid']):>3}d | {int(r['gap_hybrid']):>+4}d | On-Time Regular Refill")

    # 2. Patient Came Later (Gap / Outside Buy)
    s_later = test_df[(test_df["gap_hybrid"] <= -12) & (test_df["target_interval"] <= 60)].head(3)
    for _, r in s_later.iterrows():
        print(f"{r['customerId']:<10} | {str(r['customerName'])[:18]:<18} | {str(r['itemName'])[:24]:<24} | {int(r['target_interval']):>4}d | {int(r['pred_hybrid']):>3}d | {int(r['gap_hybrid']):>+4}d | Patient Delayed / Gap / Bought Outside")

    # 3. Patient Came Earlier (Early Top-Up / Dosage change)
    s_early = test_df[(test_df["gap_hybrid"] >= 10) & (test_df["target_interval"] <= 30)].head(3)
    for _, r in s_early.iterrows():
        print(f"{r['customerId']:<10} | {str(r['customerName'])[:18]:<18} | {str(r['itemName'])[:24]:<24} | {int(r['target_interval']):>4}d | {int(r['pred_hybrid']):>3}d | {int(r['gap_hybrid']):>+4}d | Early Top-Up / Dose Escalation")
    print("-" * 115)


if __name__ == "__main__":
    run_detailed_analysis()
