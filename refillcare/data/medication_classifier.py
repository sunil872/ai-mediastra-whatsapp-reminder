"""Medication Classifier & Chronic Therapy Gate.

Categorizes pharmacy items into Chronic Maintenance Medications vs Non-Chronic / OTC / FMCG /
Acute / Surgical / Personal Care items.

Ensures that only true chronic maintenance therapies enter predictive ML modeling and
active automated refill reminder dispatch, while preserving OTC and general items in
the database for analytics and audit reporting.
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional, Set, List, Tuple
import pandas as pd


# ------------------------------------------------------------------------------
# 1. NON-CHRONIC & OTC/FMCG EXCLUSION KEYWORDS & PATTERNS
# ------------------------------------------------------------------------------

# Exact and prefix/substring keyword categories for non-chronic items
EXCLUDED_CATEGORIES: Dict[str, List[str]] = {
    "PERSONAL_CARE_AND_HYGIENE": [
        "SOAP", "SOAPS", "FACE WASH", "FACEWASH", "FACE CREAM", "SHAMPOO",
        "HAIR OIL", "HAIR SERUM", "HAIR GEL", "HAIR COLOUR", "HAIR DYE", "OIL",
        "BUDS", "COTTON BUD", "COTTON BUDS", "COTTON", "COTTON ROLL",
        "BRUSH", "TOOTH PASTE", "TOOTHPASTE", "TOOTH BRUSH", "TOOTHBRUSH",
        "WIPES", "WIPE", "BABY WIPES", "CLEANSER", "SERUM", "MOISTURIZER",
        "SUNSCREEN", "LIP BALM", "VASELINE", "PONDS", "NIVEA", "CETAPHIL",
        "HIMALAYA BABY", "BABY LOTION", "BABY CREAM", "BABY SOAP", "BABY POWDER",
        "LOTION", "BODY LOTION", "CREAM", "COLD CREAM", "SKIN CREAM",
        "POWDER", "PWD", "DUSTING POWDER", "CANDID POWDER",
        "PERFUME", "DEODORANT", "DEO", "TALC", "TALCUM",
    ],
    "OTC_BALMS_GELS_AND_PAIN_RELIEF": [
        "VICKS", "VAPORUB", "INHALER", "OMNIGEL", "VOLINI", "MOOV", "IODEX",
        "BOROLINE", "BOROPLUS", "BORNOL", "FAST RELIEF", "AMRUTANJAN",
        "ZANDU BALM", "TIGER BALM", "PAIN BALM", "PAIN GEL", "PAIN RELIEF GEL",
        "PAIN SPRAY", "BALM", "SPRAY", "LINIMENT", "GEL", "OINTMENT",
    ],
    "ACUTE_AND_PRN_ANALGESICS": [
        "DOLO", "DOLO 650", "DOLO 650MG", "DOLO 650MG TAB", "DOLO 500", "DOLO 500MG", "DOLO 250",
        "DOLO COLD", "DOLO MF", "DOLO-650", "DOLO-500", "DOLOKIND", "DOLOKIND PLUS", "DOLOKIND MR",
        "DOLOWIN", "DOLONEX", "DOLOPAR", "DOLONEURON", "CALPOL", "CALPOL 650", "CALPOL 500",
        "CALPOL 250", "CALPOL 120", "CALPOL DROP", "CALPOL T", "PARACETAMOL", "PARACETAMOL 500",
        "PARACETAMOL 650", "DISPRIN", "SARIDON", "COMBIFLAM", "MEFTAL", "MEFTAL SPAS", "MEFTAL P",
        "CHESTON COLD", "SINAREST", "SUMO", "NICE", "NIMESULIDE", "NICE TAB", "NIZEN", "NIMPEX",
        "VOMIKIND", "ONDEM", "EMESET", "AVOMINE", "P-650", "PACIMOL", "CROCYN", "PYRICOOL",
        "FEVASTIN", "LOZENGES", "LOZ", "COUGH LOZ", "STREPSILS", "VICKS COUGH",
        "ENO", "ENO FRUIT", "GAS-O-FAST", "PUCIT", "DROPS", "ORAL DROPS",
    ],
    "PEDIATRIC_AND_KIDS_MEDICATIONS": [
        "KID", "KIDS", "KIDZ", "PED", "PEDIATRIC", "PAEDIATRIC", "BABY", "INFANT",
        "KID TAB", "KID SYP", "KID DROPS", "KID SUSP", "KID SUSPENSION", "KID TABLET",
        "KID TABLETS", "PED DROPS", "PEDIATRIC DROPS", "PAEDIATRIC DROPS",
        "KIDS NANO DROPS", "PED N/S", "KIDS PRO",
    ],
    "SURGICAL_AND_CONSUMABLES": [
        "SURGICAL", "BANDAGE", "BAND-AID", "BANDAID", "STERILE WATER", "WATER",
        "WATER 5ML", "WATER 10ML", "DISTILLED WATER", "NEEDLE", "SYRINGE",
        "GLOVES", "MASK", "N95", "DIAPER", "DIAPERS", "PAMPERS",
        "MAMY POKO", "MAMYPOKO", "WHISPER", "SANITARY PAD", "SANITARY NAPKIN",
        "SANITIZER", "HAND RUB", "HAND WASH", "HANDWASH", "GAUZE", "PLASTER",
        "THERMOMETER", "LANCET", "COTTON SWAB",
    ],
    "NUTRITION_FMCG_AND_SACHETS": [
        "SIMILAC", "PEDIASURE", "THREPTIN", "NAN PRO", "CERELAC", "HORLICKS",
        "BOOST", "ENSURE", "PROTINEX", "PROTEIN POWDER", "PROTEIN PWD", "PROTEIN",
        "ORS", "ELECTRAL", "ENERGY DRINK", "GLUCON D", "VACCINE",
        "1SAC", "SACHET", "SACHETS",
    ],
}

# Strong positive chronic therapy markers (Global standard ATC / CDSCO / FDA chronic classes)
CHRONIC_STRONG_MARKERS: List[str] = [
    # 1. Anti-Diabetic (Oral & Incretins/Insulins)
    "METFORMIN", "GLIMEPIRIDE", "GLICLAZIDE", "GLIPIZIDE", "GLIBENCLAMIDE", "GLIBOMID",
    "TENELIGLIPTIN", "SITAGLIPTIN", "VILDAGLIPTIN", "LINAGLIPTIN", "SAXAGLIPTIN", "ALOGLIPTIN",
    "DAPAGLIFLOZIN", "EMPAGLIFLOZIN", "CANAGLIFLOZIN", "REMOGLIFLOZIN",
    "VOGLIBOSE", "ACARBOSE", "PIOGLITAZONE",
    "SEMAGLUTIDE", "RYBELSUS", "OZEMPIC", "TIRZEPATIDE", "MOUNJARO", "LIRAGLUTIDE", "VICTOZA",
    "INSULIN", "LANTUS", "TRESIBA", "NOVOMIX", "RYZODEG", "HUMALOG", "MIXTARD", "DEGLUDEC", "GLARGINE", "BASALOG",
    "GLUCORYL", "GLIMISAVE", "GLIMY", "AMARYL", "INTAGLIP", "VOGLOW", "VOLIBO", "PIOZ",
    "GALVUS", "JANUVIA", "TRAJENTA", "JARDIANCE", "FORXIGA", "GLUCONORM", "DIAPRIDE",
    "ZITA", "GLYCIPHAGE", "OBIMET", "GEMER", "ZORYL", "TENALIS", "TENEPURE", "DAPAVEL", "OXRA",

    # 2. Anti-Hypertensive / Cardiovascular / Heart Failure
    "TELMISARTAN", "AMLODIPINE", "LOSARTAN", "CANDESARTAN", "VALSARTAN", "OLMESARTAN", "IRBESARTAN",
    "BISOPROLOL", "METOPROLOL", "ATENOLOL", "NEBIVOLOL", "CARVEDILOL", "PROPRANOLOL", "LABETALOL",
    "RAMIPRIL", "ENALAPRIL", "LISINOPRIL", "PERINDOPRIL",
    "CILNIDIPINE", "BENIDIPINE", "NIFEDIPINE", "FELODIPINE", "DILTIAZEM", "VERAPAMIL",
    "CHLORTHALIDONE", "HYDROCHLOROTHIAZIDE", "SPIRONOLACTONE", "EPLERENONE", "TORSEMIDE", "FUROSEMIDE",
    "SACUBITRIL", "CIDMUS", "AZMARDA", "VECARD", "TORLAC", "ALDACTONE", "DYTOR",
    "ARKAMIN", "CLONIDINE", "MINIPRESS", "PRAZOSIN", "IVABRADINE", "IVABRAD",
    "NITROGLYCERIN", "ISOSORBIDE", "SORBITRATE", "MONOTRATE", "RANOLAZINE", "RANX", "NICOSTAR", "NICORANDIL",
    "TELMA", "CONCOR", "BETALOC", "STARPRESS", "MET-XL", "METPURE", "CILACAR", "CARDACE",
    "LOSAR", "AMLONG", "AMLOKIND", "STAMLO", "BENITEC", "NEBICARD", "CARVIL", "INDERAL", "NICARDIA",

    # 3. Dyslipidemia / Cholesterol / Statin
    "ATORVASTATIN", "ROSUVASTATIN", "PITAVASTATIN", "SIMVASTATIN", "PRAVASTATIN",
    "ATORVA", "ROZAT", "ROSUVAS", "ROZAVEL", "CRESTOR", "ATORLIP", "LIPICARD",
    "LIPAGLYN", "SAROGLITAZAR", "FENOFIBRATE", "EZETIMIBE", "BEMPEDOIC ACID", "BEMPID",

    # 4. Thyroid, Hormone & Endocrine
    "LEVOTHYROXINE", "THYRONORM", "ELTROXIN", "THYROX", "THYROKAB",
    "CARBIMAZOLE", "METHIMAZOLE", "CABERGOLINE", "CABERLIN",

    # 5. Respiratory Maintenance (Chronic Inhalers & Oral Controllers)
    "BUDESONIDE", "FLUTICASONE", "FORACORT", "BUDECORT", "SERETIDE", "DUOLIN", "ASTHALIN", "AEROCORT",
    "TIOTROPIUM", "TIOMATE", "GLYCOPYRRONIUM", "FORMOTEROL", "SALMETEROL",
    "MONTELUKAST", "MONTEMAC", "MONTAIR", "DOXOFYLLINE", "DOXOLIN", "ACEBROPHYLLINE", "THEOPHYLLINE", "DERIPHYLLIN",

    # 6. Neurology, Psychiatry & Chronic Pain Management
    "LEVETIRACETAM", "DIVALPROEX", "VALPROATE", "OXETOL", "OXCARBAZEPINE", "CARBAMAZEPINE", "TEGRETOL",
    "LAMOTRIGINE", "PHENYTOIN", "EPTOIN", "CLONAZEPAM", "CLONAFIT", "CLOBAZAM", "FRISIUM",
    "ESCITALOPRAM", "NEXITO", "CILENTRA", "SEROQUEL", "QUETIAPINE", "OLANZAPINE", "RISPERIDONE",
    "AMITRIPTYLINE", "PROTHIADEN", "DULOXETINE", "DUZELA",
    "PREGABALIN", "PREGABA", "PREGABID", "GABAPENTIN", "GABAPIN", "BACLOFEN",
    "DONEPEZIL", "MEMANTINE", "SYNDOPA", "LEVODOPA", "PRAMIPEXOLE",

    # 7. Nephrology, Urology & Organ Support
    "TAMSULOSIN", "URIMAX", "FINASTERIDE", "DUTASTERIDE", "SILODOSIN", "SILODAL", "ALFUZOSIN",
    "FEBUXOSTAT", "FEBUTAZ", "ALLOPURINOL", "ZYLORIC", "REVLAMER", "SEVELAMER",
    "KETOANALOGUE", "KETOLIC", "CALCIUM ACETATE", "CALCIROL", "GEMCAL",

    # 8. Gastroenterology (Chronic Maintenance & Liver/IBD)
    "PANTOPRAZOLE", "PANTOCID", "PAN-D", "PAN D", "PAN 40", "RABEPRAZOLE", "RABLET", "RABICIP",
    "ESOMEPRAZOLE", "NEXPRO", "UDCA", "URSOCOL", "URSO", "MESALAMINE", "MESALAZINE", "MESACOL",

    # 9. Rheumatology, Autoimmune & Chronic Anti-Inflammatory
    "METHOTREXATE", "FOLITRAX", "HYDROXYCHLOROQUINE", "HCQS", "SULFASALAZINE", "SAAZ",
    "LEFLUNOMIDE", "LEFRA", "TOFACITINIB", "TOFASIL", "PREDNISOLONE", "WYSOLONE", "METHYLPREDNISOLONE", "DEFLAZACORT", "DEFCORT",

    # 10. Anti-platelet & Anticoagulant
    "ASPIRIN", "ECOSPRIN", "CLOPIDOGREL", "CLOPIVAS", "DEPLATT", "TICAGRELOR", "BRILINTA", "PRASUGREL",
    "APIXABAN", "ELIQUIS", "RIVAROXABAN", "XARELTO", "DABIGATRAN", "PRADAXA", "WARFARIN",
]

# Formulations, release modifiers, combination suffixes, and strengths (Global standard)
CHRONIC_FORM_MARKERS: List[str] = [
    # Dosage Forms
    " TAB", " TABLET", " TABLETS", " CAP", " CAPSULE", " CAPSULES", " PILL", " SOFTGEL", " STRIP", " BLISTER",

    # Modified / Sustained / Extended Release Modifiers
    " ER", " SR", " XR", " XL", " PR", " CR", " MR", " DR", " TR", " LA", " CD", " EC", " FC", " RETARD", " CHRONO",

    # Dosing Frequency & Formulation Modifiers
    " OD", " BD", " TID", " QID", " HS", " FORTE", " PLUS", " DUO", " TRIO", " MAX", " COMBO",

    # Common Combination Markers in Chronic Prescriptions
    " -H", " -AM", " -D", " -CT", " -M", " -SM", " -MT", " -TG", " -VG", " -DP",

    # Strength Indicators (Milligrams)
    " 0.1MG", " 0.2MG", " 0.3MG", " 0.5MG", " 1MG", " 2MG", " 2.5MG", " 3MG", " 4MG", " 5MG",
    " 6.25MG", " 7.5MG", " 10MG", " 12.5MG", " 15MG", " 20MG", " 25MG", " 30MG", " 40MG", " 50MG",
    " 60MG", " 75MG", " 80MG", " 90MG", " 100MG", " 120MG", " 150MG", " 160MG", " 180MG", " 200MG",
    " 250MG", " 300MG", " 400MG", " 500MG", " 600MG", " 650MG", " 800MG", " 850MG", " 1000MG",

    # Microgram Strengths (Thyroid & Inhalers)
    " 12.5MCG", " 25MCG", " 37.5MCG", " 50MCG", " 62.5MCG", " 75MCG", " 88MCG", " 100MCG",
    " 112MCG", " 125MCG", " 137MCG", " 150MCG", " 175MCG", " 200MCG", " 300MCG", " 100UG", " 50UG", " 25UG",

    # Multi-Dose Fixed-Dose Combinations
    " 5/500", " 10/500", " 50/500", " 50/1000", " 5/40", " 2.5/50", " 5/20", " 5/10", " 5/80",
    " 20/12.5", " 40/12.5", " 80/12.5", " 0.2/500", " 0.3/500", " 1/500", " 2/500",
    " M1", " M2", " M3", " M4", " F1", " F2",
]


# Compile fast regex patterns for word boundaries
_EXCLUSION_REGEX_MAP: Dict[str, List[re.Pattern]] = {}
for category, kw_list in EXCLUDED_CATEGORIES.items():
    _EXCLUSION_REGEX_MAP[category] = [
        re.compile(r"(?:\b|_)" + re.escape(kw.strip().upper()) + r"(?:\b|_)", re.IGNORECASE)
        for kw in kw_list if kw.strip()
    ]

_CHRONIC_MARKER_REGEX: List[re.Pattern] = [
    re.compile(r"(?:\b|_)" + re.escape(m.strip().upper()) + r"(?:\b|_)", re.IGNORECASE)
    for m in CHRONIC_STRONG_MARKERS
]


def classify_medication(
    item_name: str,
    packing: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify a pharmacy item as Chronic Medication vs Non-Chronic/OTC/FMCG/Acute/Surgical.

    Args:
        item_name: Full medication / item name string from ERP/POS.
        packing: Optional pack size / packaging string (e.g. '1X10', '1KG', '100ML').

    Returns:
        Dict with keys:
            - is_chronic_eligible (bool): True if eligible for predictive refill reminder.
            - category (str): CHRONIC_MEDICATION | NON_CHRONIC_OTC_FMCG | SURGICAL_CONSUMABLE | NUTRITION_SUPPLEMENT.
            - matched_rule (str): Reason or rule matched.
            - exclusion_reason (Optional[str]): Clean descriptive text if excluded.
    """
    if not item_name or not str(item_name).strip():
        return {
            "is_chronic_eligible": False,
            "category": "INVALID_ITEM",
            "matched_rule": "EMPTY_ITEM_NAME",
            "exclusion_reason": "Item name is empty or missing.",
        }

    raw_name = str(item_name).strip().upper()
    raw_pack = str(packing or "").strip().upper()

    # 1. Hard Exclusions (Pediatric/Kids formulations and Acute PRN Analgesics ALWAYS excluded)
    # Check explicit KID regex
    if re.search(r"(?:\b|_)(KID|KIDS|KIDZ|PED|PEDIATRIC|PAEDIATRIC)(?:\b|_)", raw_name, re.IGNORECASE):
        return {
            "is_chronic_eligible": False,
            "category": "PEDIATRIC_AND_KIDS_MEDICATIONS",
            "matched_rule": "PEDIATRIC_KID_KEYWORD_MATCH",
            "exclusion_reason": "Pediatric / Kid formulation (Non-chronic).",
        }

    # Check explicit DOLO / PARACETAMOL / CALPOL / Acute Analgesic regex
    if re.search(r"(?:\b|_)(DOLO|DOLO\s*650|DOLO\s*650MG|DOLO\s*500|CALPOL|PARACETAMOL|MEFTAL|MEFTAL\s*SPAS|DISPRIN|SARIDON|COMBIFLAM)(?:\b|_)", raw_name, re.IGNORECASE):
        return {
            "is_chronic_eligible": False,
            "category": "ACUTE_AND_PRN_ANALGESICS",
            "matched_rule": "ACUTE_ANALGESIC_EXCLUSION",
            "exclusion_reason": "Acute / PRN analgesic / antipyretic (Non-chronic).",
        }

    # 2. Check strong positive chronic markers
    # If item explicitly contains known chronic active molecules/brands, verify it's not topical gel/wash
    is_strong_chronic = any(p.search(raw_name) for p in _CHRONIC_MARKER_REGEX)

    # 3. Check for Non-Chronic / OTC / FMCG / Acute exclusions
    for category, pattern_list in _EXCLUSION_REGEX_MAP.items():
        for pattern in pattern_list:
            if pattern.search(raw_name):
                # If it's a strong chronic molecule but has a topical form like GEL/SPRAY/CREAM, it's non-chronic
                if is_strong_chronic and not any(topical in raw_name for topical in ["GEL", "CREAM", "OINTMENT", "SPRAY", "WASH", "LOTION", "SOAP", "INHALER"]):
                    # Tablet/capsule of chronic drug takes precedence
                    pass
                else:
                    clean_cat = category.replace("_", " ").title()
                    return {
                        "is_chronic_eligible": False,
                        "category": category,
                        "matched_rule": f"EXCLUDED_PATTERN:{pattern.pattern}",
                        "exclusion_reason": f"Non-chronic item ({clean_cat}).",
                    }

    # 3. Check packing clues (e.g. 1KG, 400GM, 1SAC, 200MDI, ML liquids without chronic markers)
    if any(p in raw_pack for p in ["1KG", "2KG", "500GM", "400GM", "200GM", "1SAC"]):
        if not is_strong_chronic:
            return {
                "is_chronic_eligible": False,
                "category": "NUTRITION_FMCG_AND_SACHETS",
                "matched_rule": f"PACKING_BULK_FMCG:{raw_pack}",
                "exclusion_reason": "Bulk FMCG / Nutrition / Sachet pack size.",
            }

    # 4. If strong chronic marker matched and passed exclusion checks
    if is_strong_chronic:
        return {
            "is_chronic_eligible": True,
            "category": "CHRONIC_MEDICATION",
            "matched_rule": "CHRONIC_MARKER_MATCH",
            "exclusion_reason": None,
        }

    # 5. Check chronic formulation markers (Tablets, Capsules, ER/SR/XR/XL, strengths)
    has_form_marker = any(marker in raw_name for marker in CHRONIC_FORM_MARKERS)
    if has_form_marker and ("SYP" not in raw_name and "DROPS" not in raw_name and "RESP" not in raw_name):
        return {
            "is_chronic_eligible": True,
            "category": "CHRONIC_MEDICATION",
            "matched_rule": "CHRONIC_FORMULATION_RULE",
            "exclusion_reason": None,
        }

    # 6. Default to standard medication if typical strip/tab packaging
    if any(p in raw_pack for p in ["1X10", "1X15", "1X20", "1X30", "1X14", "1X7", "1X28"]):
        return {
            "is_chronic_eligible": True,
            "category": "CHRONIC_MEDICATION",
            "matched_rule": "STANDARD_STRIP_PACKAGING",
            "exclusion_reason": None,
        }

    # 7. Unmatched by exclusions -> Presumed regular eligible medication
    return {
        "is_chronic_eligible": True,
        "category": "REGULAR_MEDICATION",
        "matched_rule": "NO_EXCLUSION_MATCH",
        "exclusion_reason": None,
    }


def compute_item_population_chronic_stats(
    df: pd.DataFrame,
    item_col: str = "itemName",
    customer_col: str = "customerId",
    date_col: str = "invoice_date",
) -> pd.DataFrame:
    """Compute cross-customer population consensus metrics for every medication in the dataset.

    Calculates:
    - total_buyers: Total distinct patients purchasing this item.
    - repeat_buyers: Distinct patients who purchased this item >= 2 times.
    - repeat_rate: % of patients who repurchase (high repeat rate indicates chronic maintenance).
    - median_repurchase_interval: Population median days between purchases.
    - is_population_chronic_consensus: True if behavioral consensus confirms chronic maintenance therapy.
    """
    if df is None or df.empty or item_col not in df.columns or customer_col not in df.columns:
        return pd.DataFrame()

    working = df[[customer_col, item_col, date_col]].dropna().copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.dropna(subset=[date_col]).sort_values([customer_col, item_col, date_col])

    # Distinct buyers per item
    buyer_counts = working.groupby(item_col)[customer_col].nunique().rename("total_buyers")
    
    # Purchases per customer-item
    cust_item_counts = working.groupby([item_col, customer_col]).size().reset_index(name="buy_count")
    repeat_buyers = cust_item_counts[cust_item_counts["buy_count"] >= 2].groupby(item_col)[customer_col].nunique().rename("repeat_buyers")

    # Intervals between purchases for repeat buyers
    working["prev_date"] = working.groupby([customer_col, item_col])[date_col].shift(1)
    working["interval_days"] = (working[date_col] - working["prev_date"]).dt.days
    valid_intervals = working[working["interval_days"] > 0]
    med_intervals = valid_intervals.groupby(item_col)["interval_days"].median().rename("population_median_interval")

    stats_df = pd.DataFrame(buyer_counts).join(repeat_buyers, how="left").fillna(0)
    stats_df["repeat_buyers"] = stats_df["repeat_buyers"].astype(int)
    stats_df["repeat_rate"] = stats_df["repeat_buyers"] / stats_df["total_buyers"].clip(lower=1)
    stats_df = stats_df.join(med_intervals, how="left")

    # Population Consensus Threshold:
    # >= 5 unique buyers, repeat rate >= 35%, median interval between 15 and 75 days
    is_pop_chronic = (
        (stats_df["total_buyers"] >= 5) &
        (stats_df["repeat_rate"] >= 0.35) &
        (stats_df["population_median_interval"] >= 15) &
        (stats_df["population_median_interval"] <= 75)
    )
    stats_df["is_population_chronic_consensus"] = is_pop_chronic
    return stats_df


def verify_medication_chronic_status(
    item_name: str,
    patient_history_count: int = 1,
    patient_median_interval: Optional[float] = None,
    population_repeat_rate: Optional[float] = None,
    packing: Optional[str] = None,
) -> Dict[str, Any]:
    """Multi-tiered verification combining Rule-Based Clinical Parsing + Behavioral Consensus.

    Tier 1: Clinical Brand/Salt & Formulation Rules (MedicationClassifier)
    Tier 2: Personal Patient Repurchase Cadence (>= 3 recurring purchases)
    Tier 3: Cross-Customer Population Consensus (Repeat purchase rate >= 35%)
    """
    rule_res = classify_medication(item_name, packing=packing)
    
    # If explicitly excluded by rules (e.g. soap, condom, diaper, spray) -> Always False
    if not rule_res["is_chronic_eligible"]:
        return rule_res

    # If personal history demonstrates proven longitudinal adherence
    if patient_history_count >= 3 and patient_median_interval and (15.0 <= patient_median_interval <= 65.0):
        return {
            "is_chronic_eligible": True,
            "category": "CHRONIC_BY_PERSONAL_RECURRENCE",
            "matched_rule": f"PERSONAL_HISTORY_VERIFIED (Buys: {patient_history_count}, Median: {patient_median_interval:.1f}d)",
            "exclusion_reason": None,
        }

    # If population consensus confirms repeat chronic behavior
    if population_repeat_rate is not None and population_repeat_rate >= 0.35:
        return {
            "is_chronic_eligible": True,
            "category": "CHRONIC_BY_POPULATION_CONSENSUS",
            "matched_rule": f"POPULATION_CONSENSUS_VERIFIED (Repeat Rate: {population_repeat_rate*100:.1f}%)",
            "exclusion_reason": None,
        }

    return rule_res


def is_chronic_medication(
    item_name: str,
    packing: Optional[str] = None,
) -> bool:
    """Convenience boolean helper to check if an item is eligible chronic medication."""
    res = classify_medication(item_name, packing=packing)
    return bool(res["is_chronic_eligible"])


def filter_chronic_transactions(
    df: pd.DataFrame,
    item_col: str = "itemName",
    packing_col: str = "packing",
) -> pd.DataFrame:
    """Filter a DataFrame of transactions to keep only eligible chronic medication records.

    Args:
        df: Input transactions DataFrame.
        item_col: Name of the item name column.
        packing_col: Name of the packing column if available.

    Returns:
        Filtered DataFrame containing only chronic medication purchases.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if item_col not in df.columns:
        return df.copy()

    items = df[item_col].fillna("").astype(str).tolist()
    packings = df[packing_col].fillna("").astype(str).tolist() if packing_col in df.columns else [None] * len(df)

    eligible_mask = [
        is_chronic_medication(item, pack)
        for item, pack in zip(items, packings)
    ]

    return pd.DataFrame(df[pd.Series(eligible_mask, index=df.index)].copy())

