"""Medication Classifier & Chronic Therapy Gate.

Categorizes pharmacy items into Chronic Maintenance Medications vs Non-Chronic / OTC / FMCG /
Acute / Surgical / Personal Care items.

Ensures that only true chronic maintenance therapies enter predictive ML modeling and
active automated refill reminder dispatch, while preserving OTC and general items in
the database for analytics and audit reporting.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Any, Optional, Set, List, Tuple
import pandas as pd


# ------------------------------------------------------------------------------
# 1. NON-CHRONIC & OTC/FMCG EXCLUSION KEYWORDS & PATTERNS
# ------------------------------------------------------------------------------

EXCLUDED_CATEGORIES: Dict[str, List[str]] = {
    "FORMULATION_SYRUPS_AND_SUSPENSIONS": [
        "SYP", "SYRUP", "SUSP", "SUSPENSION", "ELIXIR", "LIQ", "LIQUID", "SYP.", "SUSP.",
        "DRY SYRUP", "ORAL SUSPENSION", "ORAL SOLUTION", "EXPECTORANT", "LINCTUS", "TONIC", "APPETITE TONIC",
    ],
    "FORMULATION_DROPS_OPHTHALMIC_OTIC": [
        "DROP", "DROPS", "EYE DROP", "EYE DROPS", "EAR DROP", "EAR DROPS", "NASAL DROP", "NASAL DROPS",
        "ORAL DROP", "ORAL DROPS", "PEDIATRIC DROPS", "PAEDIATRIC DROPS", "OPHTH", "OPHTHALMIC", "OTIC",
        "DRPOS", "DPR", "DRP", "EYE/EAR DROP", "EYE/EAR DRPOS",
        "SODIUM HYALURONATE", "HYPROMELLOSE", "NEPAFENAC", "CARBOXYMETHYLCELLULOSE",
        "TIMOLOL", "BRIMONIDINE", "DORZOLAMIDE", "LATANOPROST", "BIMATOPROST", "TRAVAPROST",
        "TRAVOPROST", "OLOPATADINE", "REFRESH TEARS", "SYSTANE", "LUBREX", "ALPHAGAN", "DORZOX",
    ],
    "FORMULATION_RESPULES_AND_NEBULIZERS": [
        "RESPULE", "RESPULES", "RESP", "RESP.", "NEBU", "NEBULIZER", "NEBULISER",
        "NEB", "NEB.", "NEBULES", "MISTPULES", "SMARTULES", "INHALATION SOLUTION",
        "BUDECORT RESPULES", "DUOLIN RESPULES", "FORACORT RESPULES", "LEVOLIN RESPULES",
        "ASTHALIN RESPULES", "IPRATROPIUM RESPULES", "BUDESONIDE RESPULES", "SALBAIR RESP",
        "COMBIMIST RESPULES", "FORMONIDE RESPULES", "NEBZMART RESPULES",
    ],
    "FORMULATION_TOPICALS_DERMA": [
        "OINT", "OINTMENT", "CREAM", "CRM", "GEL", "EMULGEL", "LOTION", "LINIMENT", "SOAP", "SHAMPOO", "WASH", "FACE WASH", "PASTE",
    ],
    "FORMULATION_SPRAYS": [
        "SPRAY", "NASAL SPRAY", "PAIN SPRAY",
    ],
    "FORMULATION_INJECTIONS": [
        "INJ", "INJECTION", "INFUSION", "IV", "IM", "DEPOT",
    ],
    "FORMULATION_SACHETS_GRANULES": [
        "SACHET", "SACHETS", "GRANULES", "1SAC",
    ],
    "ACUTE_ANTIBIOTICS": [
        "AUGMENTIN", "CLAVAM", "MOXIKIND", "TAXIM", "MONOCEF", "DOXT", "BACTRIM", "AZITHRAL", "AZITHROMYCIN",
        "CIPROFLOXACIN", "CIPLOX", "OFLOXACIN", "OFLOX", "NORFLOXACIN", "NORFLOX", "METROGYL", "METRONIDAZOLE",
        "CEFIXIME", "MAHACEF", "ROXID", "ZADY", "AMPOXIN", "CEFPODOXIME", "CEFUROXIME", "CEFTUM", "FAROPENEM",
        "FAROBACT", "FARONEM", "MEROPENEM", "LINEZOLID", "LIZOMAC", "DOXYCYCLINE", "NITROFURANTOIN", "NIFTRAN",
        "CO-TRIMOXAZOLE", "SEPTRA", "AMOXICILLIN", "AMOXYCILLIN", "AMOXICLAV", "CLAVULANATE", "CLAVULANIC ACID",
    ],
    "ACUTE_AND_PRN_ANALGESICS": [
        "DOLO", "DOLO 650", "DOLO 650MG", "DOLO 650MG TAB", "DOLO 500", "DOLO 500MG", "DOLO 250",
        "DOLO COLD", "DOLO MF", "DOLO-650", "DOLO-500", "DOLOKIND", "DOLOKIND PLUS", "DOLOKIND MR",
        "DOLOWIN", "DOLONEX", "DOLOPAR", "DOLONEURON", "CALPOL", "CALPOL 650", "CALPOL 500",
        "CALPOL 250", "CALPOL 120", "CALPOL DROP", "CALPOL T", "PARACETAMOL", "PARACETAMOL 500",
        "PARACETAMOL 650", "DISPRIN", "SARIDON", "COMBIFLAM", "MEFTAL", "MEFTAL SPAS", "MEFTAL-SPAS", "MEFTAL P",
        "CHESTON COLD", "SINAREST", "SUMO", "NICE", "NIMESULIDE", "NICE TAB", "NIZEN", "NIMPEX",
        "VOMIKIND", "ONDEM", "EMESET", "AVOMINE", "P-650", "PACIMOL", "CROCYN", "PYRICOOL",
        "FEVASTIN", "LOZENGES", "LOZ", "COUGH LOZ", "STREPSILS", "VICKS COUGH",
        "ENO", "ENO FRUIT", "GAS-O-FAST", "PUCIT", "DROPS", "ORAL DROPS",
        "KETOROL", "KETOROL DT", "DART", "HEADSET", "ZERODOL SP", "ZERODOL-SP", "ZERODOL P", "ZERODOL-P",
        "FLEXON", "FLEXON MR", "ULTRACET", "PARACIP", "BRUFEN", "IBUGESIC", "TRAMADOL", "TRAMAZAC",
    ],
    "ACUTE_COUGH_AND_COLD": [
        "SOLVIN COLD", "FEBREX PLUS", "CHESTON COLD", "SINAREST", "ALERID-D", "ALERID D", "WIKORYL", "FLUDREX", "LEVOCOLD",
    ],
    "ACUTE_ANTACIDS_AND_DIARRHEA": [
        "ELDOPER", "LOPERAMIDE", "LOPO", "ENO", "PUDIN HARA", "DIGENE", "GELUSIL", "GAVISCON", "SUCRAFIL", "MUCAINE", "CREMAFFIN",
        "PAN-D", "PAN D", "PAN 40", "PANTOCID D", "PANTODAC D", "RABEKIND DSR", "RABEKIND-DSR", "RABEKIND",
        "RABLET-D", "RABLET D", "OCID-D", "OCID D", "OMEE D", "ESOKEM-D", "DOMPAN", "DULCOFLEX", "BISACODYL",
    ],
    "LOZENGES_AND_OTC_CONFECTIONS": [
        "COFSILS", "STREPSILS", "VICKS", "VICS", "HONITUS",
    ],
    "ACUTE_ANTIFUNGAL_TOPICAL": [
        "LULIFORD", "CANESTEN", "CLOCIP", "CANDID", "ITMAC", "ITRASYS", "ZOCON",
    ],
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
    "GLYCOMET", "GLIMESTAR", "GLIMIPRIME", "CYBLEX", "UDAPA", "GLUCRETA", "OBORAVO", "ZOMELIS", "GLUCOBAY", "SITAXA", "PRACTOMET",

    # 2. Anti-Hypertensive / Cardiovascular / Heart Failure
    "TELMISARTAN", "AMLODIPINE", "LOSARTAN", "CANDESARTAN", "VALSARTAN", "OLMESARTAN", "IRBESARTAN",
    "BISOPROLOL", "METOPROLOL", "ATENOLOL", "NEBIVOLOL", "CARVEDILOL", "PROPRANOLOL", "LABETALOL",
    "RAMIPRIL", "ENALAPRIL", "LISINOPRIL", "PERINDOPRIL",
    "CILNIDIPINE", "BENIDIPINE", "NIFEDIPINE", "FELODIPINE", "DILTIAZEM", "VERAPAMIL",
    "CHLORTHALIDONE", "HYDROCHLOROTHIAZIDE", "SPIRONOLACTONE", "EPLERENONE", "TORSEMIDE", "FUROSEMIDE",
    "SACUBITRIL", "CIDMUS", "AZMARDA", "VECARD", "TORLAC", "ALDACTONE", "DYTOR",
    "ARKAMIN", "CLONIDINE", "MINIPRESS", "PRAZOSIN", "IVABRADINE", "IVABRAD", "VERICIGUAT",
    "NITROGLYCERIN", "GLYCERYL TRINITRATE", "ISOSORBIDE", "SORBITRATE", "MONOTRATE", "RANOLAZINE", "RANX", "NICOSTAR", "NICORANDIL",
    "TELMA", "CONCOR", "BETALOC", "STARPRESS", "MET-XL", "METPURE", "CILACAR", "CARDACE",
    "LOSAR", "AMLONG", "AMLOKIND", "STAMLO", "BENITEC", "NEBICARD", "CARVIL", "INDERAL", "NICARDIA",
    "ERITEL", "TELMIKIND", "TELSAR", "TELISTA", "CORTEL", "TELLZY", "CARDIVAS", "CTD",

    # 3. Dyslipidemia / Cholesterol / Statin
    "ATORVASTATIN", "ROSUVASTATIN", "PITAVASTATIN", "SIMVASTATIN", "PRAVASTATIN",
    "ATORVA", "ROZAT", "ROSUVAS", "ROZAVEL", "CRESTOR", "ATORLIP", "LIPICARD",
    "LIPAGLYN", "SAROGLITAZAR", "FENOFIBRATE", "EZETIMIBE", "BEMPEDOIC ACID", "BEMPID",
    "ROSEDAY", "RAZEL", "ROZUCOR", "TONACT", "ROSYCAP",

    # 4. Thyroid, Hormone & Endocrine
    "LEVOTHYROXINE", "THYRONORM", "ELTROXIN", "THYROX", "THYROKAB",
    "CARBIMAZOLE", "METHIMAZOLE", "CABERGOLINE", "CABERLIN",

    # 5. Respiratory Maintenance (Chronic Inhalers & Oral Controllers)
    "BUDESONIDE", "FLUTICASONE", "FORACORT", "BUDECORT", "SERETIDE", "DUOLIN", "ASTHALIN", "AEROCORT",
    "TIOTROPIUM", "TIOMATE", "GLYCOPYRRONIUM", "FORMOTEROL", "SALMETEROL",
    "MONTELUKAST", "MONTEMAC", "MONTAIR", "DOXOFYLLINE", "DOXOLIN", "ACEBROPHYLLINE", "THEOPHYLLINE", "DERIPHYLLIN",

    # 6. Neurology, Psychiatry & Chronic Pain Management
    "LEVETIRACETAM", "DIVALPROEX", "VALPROATE", "VALPROIC ACID", "OXETOL", "OXCARBAZEPINE", "CARBAMAZEPINE", "TEGRETOL",
    "LAMOTRIGINE", "PHENYTOIN", "EPTOIN", "CLONAZEPAM", "CLONAFIT", "CLOBAZAM", "FRISIUM",
    "ESCITALOPRAM", "NEXITO", "CILENTRA", "SEROQUEL", "QUETIAPINE", "OLANZAPINE", "RISPERIDONE",
    "AMITRIPTYLINE", "PROTHIADEN", "DULOXETINE", "DUZELA",
    "PREGABALIN", "PREGABA", "PREGABID", "GABAPENTIN", "GABAPIN", "BACLOFEN",
    "DONEPEZIL", "MEMANTINE", "SYNDOPA", "LEVODOPA", "PRAMIPEXOLE", "FLUNARIZINE",

    # 7. Nephrology, Urology & Organ Support
    "TAMSULOSIN", "URIMAX", "FINASTERIDE", "DUTASTERIDE", "SILODOSIN", "SILODAL", "ALFUZOSIN",
    "FEBUXOSTAT", "FEBUTAZ", "ALLOPURINOL", "ZYLORIC", "REVLAMER", "SEVELAMER",
    "KETOANALOGUE", "KETOLIC", "CALCIUM ACETATE", "CALCIROL", "GEMCAL",

    # 8. Gastroenterology (Chronic Maintenance & Liver/IBD)
    "PANTOPRAZOLE", "PANTOCID", "RABEPRAZOLE", "RABLET", "RABICIP",
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
    " ER", " SR", " XR", " XL", " PR", " CR", " MR", " -DR", " TR", " LA", " CD", " EC", " FC", " RETARD", " CHRONO",

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


import os
import time
from pathlib import Path

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

# ------------------------------------------------------------------------------
# CHEMICAL ACTIVE INGREDIENT / SALT MOLECULE DEFINITIONS
# ------------------------------------------------------------------------------

CHRONIC_SALT_MOLECULES: Set[str] = {
    # Anti-Diabetic (Oral & Insulin)
    "METFORMIN", "GLIMEPIRIDE", "GLICLAZIDE", "GLIPIZIDE", "GLIBENCLAMIDE", "GLIBOMID",
    "TENELIGLIPTIN", "SITAGLIPTIN", "VILDAGLIPTIN", "LINAGLIPTIN", "SAXAGLIPTIN", "ALOGLIPTIN",
    "DAPAGLIFLOZIN", "EMPAGLIFLOZIN", "CANAGLIFLOZIN", "REMOGLIFLOZIN",
    "VOGLIBOSE", "ACARBOSE", "PIOGLITAZONE",
    "SEMAGLUTIDE", "TIRZEPATIDE", "LIRAGLUTIDE",
    "INSULIN", "INSULIN GLARGINE", "INSULIN DEGLUDEC", "INSULIN ASPART", "INSULIN LISPRO", "BIPHASIC ISOPHANE INSULIN",

    # Cardiovascular & Anti-Hypertensive
    "TELMISARTAN", "AMLODIPINE", "LOSARTAN", "CANDESARTAN", "VALSARTAN", "OLMESARTAN", "IRBESARTAN",
    "BISOPROLOL", "METOPROLOL", "ATENOLOL", "NEBIVOLOL", "CARVEDILOL", "PROPRANOLOL", "LABETALOL",
    "RAMIPRIL", "ENALAPRIL", "LISINOPRIL", "PERINDOPRIL",
    "CILNIDIPINE", "BENIDIPINE", "NIFEDIPINE", "FELODIPINE", "DILTIAZEM", "VERAPAMIL",
    "CHLORTHALIDONE", "HYDROCHLOROTHIAZIDE", "SPIRONOLACTONE", "EPLERENONE", "TORSEMIDE", "FUROSEMIDE",
    "SACUBITRIL", "IVABRADINE", "NICORANDIL", "NITROGLYCERIN", "GLYCERYL TRINITRATE", "ISOSORBIDE",
    "RANOLAZINE", "CLONIDINE", "PRAZOSIN", "VERICIGUAT",

    # Lipid & Cholesterol
    "ATORVASTATIN", "ROSUVASTATIN", "PITAVASTATIN", "SIMVASTATIN", "PRAVASTATIN",
    "FENOFIBRATE", "EZETIMIBE", "BEMPEDOIC ACID", "SAROGLITAZAR",

    # Thyroid & Endocrine
    "LEVOTHYROXINE", "LEVOTHYROXINE SODIUM", "THYROXINE", "CARBIMAZOLE", "METHIMAZOLE", "CABERGOLINE",

    # Respiratory Maintenance Controllers
    "BUDESONIDE", "FLUTICASONE", "TIOTROPIUM", "GLYCOPYRRONIUM", "FORMOTEROL", "SALMETEROL",
    "MONTELUKAST", "DOXOFYLLINE", "THEOPHYLLINE", "ACEBROPHYLLINE",

    # Neurology, Psychiatry & Neuropathic Pain
    "LEVETIRACETAM", "DIVALPROEX", "VALPROATE", "VALPROIC ACID", "SODIUM VALPROATE", "OXCARBAZEPINE", "CARBAMAZEPINE",
    "LAMOTRIGINE", "PHENYTOIN", "CLONAZEPAM", "CLOBAZAM",
    "ESCITALOPRAM", "SERTRALINE", "PAROXETINE", "FLUVOXAMINE", "FLUOXETINE", "DULOXETINE", "AMITRIPTYLINE",
    "QUETIAPINE", "OLANZAPINE", "RISPERIDONE",
    "PREGABALIN", "GABAPENTIN", "BACLOFEN", "DONEPEZIL", "MEMANTINE", "LEVODOPA", "CARBIDOPA", "PRAMIPEXOLE", "FLUNARIZINE",

    # Urology, Nephrology & Organ Support
    "TAMSULOSIN", "SILODOSIN", "ALFUZOSIN", "FINASTERIDE", "DUTASTERIDE",
    "FEBUXOSTAT", "ALLOPURINOL", "SEVELAMER", "CALCIUM ACETATE", "KETOANALOGUE",

    # Chronic GI Maintenance
    "URSODEOXYCHOLIC ACID", "UDCA", "MESALAMINE", "MESALAZINE", "SULFASALAZINE",

    # Rheumatology & Autoimmune
    "METHOTREXATE", "HYDROXYCHLOROQUINE", "LEFLUNOMIDE", "TOFACITINIB",

    # Anti-platelet & Anticoagulant
    "ASPIRIN", "CLOPIDOGREL", "TICAGRELOR", "PRASUGREL", "APIXABAN", "RIVAROXABAN", "DABIGATRAN", "WARFARIN",
}

ACUTE_EXCLUDED_SALT_MOLECULES: Set[str] = {
    # Acute Antibiotics & Antimicrobials
    "AMOXICILLIN", "AMOXYCILLIN", "CLAVULANIC ACID", "AZITHROMYCIN", "CIPROFLOXACIN", "OFLOXACIN",
    "NORFLOXACIN", "LEVOFLOXACIN", "CEFIXIME", "CEFPODOXIME", "CEFUROXIME", "LINEZOLID", "FAROPENEM",
    "MEROPENEM", "METRONIDAZOLE", "ROXITHROMYCIN", "DOXYCYCLINE", "NITROFURANTOIN", "CO-TRIMOXAZOLE", "BACTRIM",

    # Acute Analgesics & Antipyretics
    "PARACETAMOL", "ACECLOFENAC", "DICLOFENAC", "IBUPROFEN", "NIMESULIDE", "TRAMADOL", "KETOROLAC",
    "PIROXICAM", "NAPROXEN", "MEFENAMIC ACID",

    # Acute GI & Antacids (and PPI combinations with prokinetics)
    "DOMPERIDONE", "ONDANSETRON", "LOPERAMIDE", "RACECADOTRIL", "SUCRALFATE", "ALUMINIUM HYDROXIDE",
    "MAGNESIUM HYDROXIDE", "SIMETHICONE", "LACTULOSE", "BISACODYL", "SENNA", "PICOSULFATE", "DOCUSATE",

    # Acute Cough, Cold & Allergies
    "DEXTROMETHORPHAN", "CHLORPHENIRAMINE", "PHENYLEPHRINE", "AMBROXOL", "BROMHEXINE", "GUAIFENESIN",
    "TERBUTALINE", "LEVOCETIRIZINE", "CETIRIZINE", "FEXOFENADINE", "BILASTINE", "CYPROHEPTADINE",

    # Anti-parasitic & Anthelmintic
    "ALBENDAZOLE", "IVERMECTIN", "MEBENDAZOLE", "DIETHYLCARBAMAZINE",

    # Acute & Topical Antifungals
    "FLUCONAZOLE", "ITRACONAZOLE", "LULICONAZOLE", "CLOBETASOL", "KETOCONAZOLE", "AMOROLFINE", "TERBINAFINE",

    # Ophthalmic & Otic (Drops)
    "SODIUM HYALURONATE", "HYPROMELLOSE", "NEPAFENAC", "MOXIFLOXACIN", "TOBRAMYCIN", "GENTAMICIN",
    "CARBOXYMETHYLCELLULOSE", "TIMOLOL", "BRIMONIDINE", "DORZOLAMIDE", "LATANOPROST", "BIMATOPROST", "TRAVAPROST",
}

# Master SALT Catalog In-Memory Cache (Item Name -> Salt, Category, Pack)
_SALT_CATALOG: Dict[str, Dict[str, str]] = {}
_SALT_CATALOG_LOADED: bool = False


def extract_salt_molecules(salt_str: Optional[str]) -> List[str]:
    """Extract individual active molecule names from a chemical composition string."""
    if not salt_str or not str(salt_str).strip() or str(salt_str).strip() in ("[00]", "NONE", "NAN"):
        return []
    raw = str(salt_str).strip().upper()
    raw = re.sub(r"\[\s*\d+\s*\]", " ", raw)
    parts = re.split(r"[+&/,]", raw)
    molecules: List[str] = []
    for p in parts:
        cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:MG|MCG|GM|G|IU|%|W/V|W/W|ML|UNITS|UNIT)\b", " ", p, flags=re.IGNORECASE)
        cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
        cleaned = re.sub(r"[-_](?:PR|SR|ER|CR|XL|MR|DT|IR)\b", " ", cleaned)
        cleaned = re.sub(r"[^A-Z\s]", " ", cleaned)
        words = [w.strip() for w in cleaned.split() if len(w.strip()) >= 3 and w.strip() not in ("TAB", "CAP", "INJ", "SYP", "SUSP", "FOR", "AND", "WITH", "THE")]
        if words:
            full_mol = " ".join(words)
            molecules.append(full_mol)
            for w in words:
                if len(w) >= 4 and w != full_mol:
                    molecules.append(w)
    return list(dict.fromkeys(molecules))


def load_salt_catalog() -> Dict[str, Dict[str, str]]:
    """Load item-to-salt catalog from SALT WISE ITEMS.csv (or .xlsx) with in-memory caching."""
    global _SALT_CATALOG, _SALT_CATALOG_LOADED
    if _SALT_CATALOG_LOADED:
        return _SALT_CATALOG

    import csv
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_candidates = [
        base_dir / "data" / "refillcare" / "SALT WISE ITEMS.csv",
        Path("data/refillcare/SALT WISE ITEMS.csv"),
    ]

    for cpath in csv_candidates:
        if cpath.exists():
            try:
                with open(cpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("S.No,"):
                            header = [c.strip() for c in line.split(",")]
                            break
                    reader = csv.DictReader(f, fieldnames=header)
                    for row in reader:
                        name = row.get("Item Name", "").strip().upper()
                        salt = row.get("SALT", "").strip().upper()
                        cat = row.get("CATEGORY", "").strip().upper()
                        pack = row.get("PACK", "").strip().upper()
                        icat = row.get("ITEMCAT", "").strip().upper()
                        if name:
                            _SALT_CATALOG[name] = {
                                "salt": salt if salt and salt != "[00]" else "",
                                "category": cat,
                                "pack": pack,
                                "itemcat": icat,
                            }
                _SALT_CATALOG_LOADED = True
                return _SALT_CATALOG
            except Exception:
                pass

    _SALT_CATALOG_LOADED = True
    return _SALT_CATALOG


def get_catalog_item_salt(item_name: str) -> Optional[Dict[str, str]]:
    """Look up item salt and category composition from master catalog."""
    if not item_name:
        return None
    catalog = load_salt_catalog()
    clean_name = str(item_name).strip().upper()
    return catalog.get(clean_name)


# ------------------------------------------------------------------------------
# SUPABASE DYNAMIC CLASSIFICATION RULES PROVIDER & CACHE
# ------------------------------------------------------------------------------
_SUPABASE_CLIENT = None
_CACHED_RULES_TIMESTAMP: float = 0.0
_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes in-memory TTL
# Negative cache: after a failed / empty Supabase fetch, do not retry on every classification
# call (an outage previously cost ~2 s per item). Retry at most once per interval.
_FAILURE_RETRY_SECONDS: float = float(os.getenv("SUPABASE_RULES_RETRY_SECONDS", "60"))
_LAST_FETCH_ATTEMPT: float = 0.0
_RULES_SOURCE: str = "builtin"  # builtin | supabase

_DYNAMIC_EXCLUSIONS: Set[str] = set()
_DYNAMIC_CHRONIC: Set[str] = set()
_DYNAMIC_EXCLUSION_REGEX: List[re.Pattern] = []
_DYNAMIC_CHRONIC_REGEX: List[re.Pattern] = []


def get_supabase_client():
    """Retrieve or initialize Supabase client safely."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if supabase_url and supabase_key:
        try:
            from supabase import create_client
            _SUPABASE_CLIENT = create_client(supabase_url, supabase_key)
            return _SUPABASE_CLIENT
        except Exception:
            return None
    return None


def _reset_fetch_throttle() -> None:
    global _LAST_FETCH_ATTEMPT
    _LAST_FETCH_ATTEMPT = 0.0


def clear_classification_caches() -> None:
    """Drop memoised classification results (call whenever the rule set changes)."""
    for fn_name in ("is_chronic_medication", "classify_medication_cached"):
        fn = globals().get(fn_name)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


def get_rules_status() -> Dict[str, Any]:
    """Diagnostics for /health: where classification rules currently come from."""
    return {
        "source": _RULES_SOURCE,
        "dynamic_exclusions": len(_DYNAMIC_EXCLUSIONS),
        "dynamic_chronic": len(_DYNAMIC_CHRONIC),
        "last_success_epoch": _CACHED_RULES_TIMESTAMP or None,
        "last_attempt_epoch": _LAST_FETCH_ATTEMPT or None,
    }


def get_dynamic_classification_rules() -> Tuple[Set[str], Set[str]]:
    """Fetch active dynamic classification rules from Supabase with full pagination and local fallback."""
    global _CACHED_RULES_TIMESTAMP, _DYNAMIC_EXCLUSIONS, _DYNAMIC_CHRONIC, _DYNAMIC_EXCLUSION_REGEX, _DYNAMIC_CHRONIC_REGEX
    global _LAST_FETCH_ATTEMPT, _RULES_SOURCE
    now = time.time()
    if (now - _CACHED_RULES_TIMESTAMP) < _CACHE_TTL_SECONDS and (_DYNAMIC_EXCLUSIONS or _DYNAMIC_CHRONIC):
        return _DYNAMIC_EXCLUSIONS, _DYNAMIC_CHRONIC
    if _LAST_FETCH_ATTEMPT > 0 and (now - _LAST_FETCH_ATTEMPT) < _FAILURE_RETRY_SECONDS:
        # A fetch was attempted recently (failed, empty, or just refreshed): serve last known rules.
        return _DYNAMIC_EXCLUSIONS, _DYNAMIC_CHRONIC
    _LAST_FETCH_ATTEMPT = now

    client = get_supabase_client()
    if client:
        try:
            all_rows: List[Dict[str, Any]] = []
            page_size = 1000
            offset = 0
            while True:
                res = (
                    client.table("medication_classification_rules")
                    .select("medication_name_or_keyword,rule_type")
                    .eq("is_active", True)
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                data = res.data or []
                if not data:
                    break
                all_rows.extend(data)
                if len(data) < page_size:
                    break
                offset += len(data)

            if all_rows:
                excl_set = set()
                chr_set = set()
                for row in all_rows:
                    kw = str(row.get("medication_name_or_keyword", "")).strip().upper()
                    r_type = str(row.get("rule_type", "")).strip().upper()
                    if kw:
                        if r_type == "EXCLUDE":
                            excl_set.add(kw)
                        elif r_type == "CHRONIC":
                            chr_set.add(kw)

                def _compile_rule_regex(k: str) -> re.Pattern:
                    esc = re.escape(k)
                    if re.search(r"\d+$", k):
                        return re.compile(r"(?:\b|_)" + esc + r"(?:MG|ML|MCG|GM|G|IU)?(?:\b|_)", re.IGNORECASE)
                    return re.compile(r"(?:\b|_)" + esc + r"(?:\b|_)", re.IGNORECASE)

                if excl_set != _DYNAMIC_EXCLUSIONS or chr_set != _DYNAMIC_CHRONIC:
                    clear_classification_caches()
                _RULES_SOURCE = "supabase"
                _DYNAMIC_EXCLUSIONS = excl_set
                _DYNAMIC_CHRONIC = chr_set
                _DYNAMIC_EXCLUSION_REGEX = [_compile_rule_regex(k) for k in excl_set]
                _DYNAMIC_CHRONIC_REGEX = [_compile_rule_regex(k) for k in chr_set]
                _CACHED_RULES_TIMESTAMP = now
                return _DYNAMIC_EXCLUSIONS, _DYNAMIC_CHRONIC
        except Exception:
            pass

    return _DYNAMIC_EXCLUSIONS, _DYNAMIC_CHRONIC


def add_classification_rule(
    medication_name_or_keyword: str,
    rule_type: str,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Add or update a classification rule in Supabase."""
    global _CACHED_RULES_TIMESTAMP
    client = get_supabase_client()
    if not client:
        return {"status": "error", "message": "Supabase client not configured or unreachable."}

    clean_kw = str(medication_name_or_keyword).strip().upper()
    clean_type = str(rule_type).strip().upper()
    if clean_type not in ("CHRONIC", "EXCLUDE"):
        return {"status": "error", "message": "rule_type must be either 'CHRONIC' or 'EXCLUDE'."}
    if not clean_kw:
        return {"status": "error", "message": "Medication keyword cannot be empty."}

    try:
        data = {
            "medication_name_or_keyword": clean_kw,
            "rule_type": clean_type,
            "notes": notes or "Added via UI by Pharmacist",
            "is_active": True,
        }
        res = client.table("medication_classification_rules").upsert(
            data,
            on_conflict="medication_name_or_keyword,rule_type"
        ).execute()
        _CACHED_RULES_TIMESTAMP = 0.0  # Invalidate cache for immediate sync
        _reset_fetch_throttle()
        clear_classification_caches()
        get_dynamic_classification_rules()
        return {"status": "success", "message": f"Rule for '{clean_kw}' saved as {clean_type}.", "data": res.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_classification_rules(
    search_query: str = "",
    rule_type_filter: str = "All",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieve classification rules from Supabase (paginated when limit is None to return all records)."""
    client = get_supabase_client()
    if not client:
        return []
    try:
        q = client.table("medication_classification_rules").select("*").order("created_at", desc=True)
        if rule_type_filter in ("CHRONIC", "EXCLUDE"):
            q = q.eq("rule_type", rule_type_filter)
        if search_query.strip():
            clean_s = search_query.strip().upper()
            q = q.ilike("medication_name_or_keyword", f"%{clean_s}%")
        
        if limit is not None and limit > 0:
            res = q.limit(limit).execute()
            return res.data or []
        
        # Paginated retrieval to ensure complete list is returned
        all_rules: List[Dict[str, Any]] = []
        page_size = 1000
        offset = 0
        while True:
            res = q.range(offset, offset + page_size - 1).execute()
            data = res.data or []
            if not data:
                break
            all_rules.extend(data)
            if len(data) < page_size:
                break
            offset += len(data)
        return all_rules
    except Exception:
        return []


def delete_classification_rule(rule_id: str) -> Dict[str, Any]:
    """Delete a classification rule from Supabase."""
    global _CACHED_RULES_TIMESTAMP
    client = get_supabase_client()
    if not client:
        return {"status": "error", "message": "Supabase client not configured."}
    try:
        res = client.table("medication_classification_rules").delete().eq("id", rule_id).execute()
        _CACHED_RULES_TIMESTAMP = 0.0
        _reset_fetch_throttle()
        clear_classification_caches()
        get_dynamic_classification_rules()
        return {"status": "success", "message": "Rule deleted successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def classify_medication(
    item_name: str,
    packing: Optional[str] = None,
    salt_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify a pharmacy item as Chronic Medication vs Non-Chronic/OTC/FMCG/Acute/Surgical.

    Dual-Key Clinical Architecture:
    Evaluates both Item/Brand Name and Active Salt/Chemical Composition.
    If salt_name is not provided, automatically resolves it from the SALT WISE catalog.

    Args:
        item_name: Full medication / item name string from ERP/POS.
        packing: Optional pack size / packaging string (e.g. '1X10', '1KG', '100ML').
        salt_name: Optional chemical composition / active salt string (e.g. 'EMPAGLIFLOZIN-25MG').

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
    raw_salt = str(salt_name or "").strip().upper()

    # Automatic catalog fallback if salt_name was not explicitly supplied
    catalog_cat = ""
    if not raw_salt or raw_salt in ("[00]", "NONE", "NAN"):
        cat_info = get_catalog_item_salt(raw_name)
        if cat_info:
            raw_salt = cat_info.get("salt", "")
            catalog_cat = cat_info.get("category", "")
            if not raw_pack:
                raw_pack = cat_info.get("pack", "")

    molecules = extract_salt_molecules(raw_salt)
    eval_composite = f"{raw_name} {raw_pack} {raw_salt} {catalog_cat}"

    # Load dynamic rules from Supabase (cached in memory)
    get_dynamic_classification_rules()

    # 1. Device Clinical Exemptions (Insulin Pens & Respiratory Inhalers)
    is_insulin_device = any(ins in eval_composite for ins in [
        "BASUGINE", "TRESIBA", "LANTUS", "NOVOMIX", "RYZODEG", "HUMALOG", "FIASP", "TOUJEO",
        "APIDRA", "INSULATARD", "MIXTARD", "INSULIN GLARGINE", "INSULIN DEGLUDEC", "INSULIN ASPART", "INSULIN LISPRO", "INSULIN"
    ]) and any(dev in raw_name or dev in raw_pack for dev in [
        "PEN", "CATRIAGE", "CARTRIDGE", "FLEX", "SOLOSTAR", "KWIKPEN", "PENFILL", "3ML", "1PC", "1PEN"
    ])

    is_inhaler_device = any(inh in raw_name or inh in raw_pack for inh in [
        "INHALER", "EVOHALER", "SYNCHROBREATHE", "TURBUHALER", "ACCUHALER", "AEROHALER", "ROTACAP", "RESPICAP", "200MDI", "120MD"
    ]) and not any(ex_inh in raw_name for ex_inh in [
        "NASAL SPRAY", "NASAL", "RESP", "RESPULES", "RESPULE", "RESPULS", "RESPUELS", "NEB", "NEBU", "MISTPULES", "SMARTULES", "NEBULES"
    ])

    # 2. Strict Formulation & Route Safety Gate (Syrups, Suspensions, Drops, Eye Drops, Respules, Topicals, Sprays, Sachets)
    # Mandate: All liquid/drop/syrup/respule/topical formulations are strictly suppressed from automated refill reminders
    is_non_solid = bool(re.search(
        r"(?:\b|_)(?:SYP|SYRUP|SUSP|SUSPENSION|ELIXIR|LIQ|LIQUID|DRY\s*SYRUP|ORAL\s*SUSPENSION|ORAL\s*SOLUTION|EXPECTORANT|LINCTUS|"
        r"DROP|DROPS|DRPOS|DPR|DRP|EYE\s*DROP|EYE\s*DROPS|EAR\s*DROP|EAR\s*DROPS|NASAL\s*DROP|NASAL\s*DROPS|ORAL\s*DROP|ORAL\s*DROPS|OPHTH|OPHTHALMIC|OTIC|"
        r"OINT|OINTMENT|CREAM|CRM|GEL|EMULGEL|LOTION|LINIMENT|SOAP|SHAMPOO|WASH|FACE\s*WASH|PASTE|"
        r"RESPULE|RESPULES|RESPULS|RESPUELS|RESP|NEB|NEBU|NEBULES|NEBULIZER|NEBULISER|MISTPULES|SMARTULES|INHALATION\s*SOLUTION|"
        r"SPRAY|NASAL\s*SPRAY|PAIN\s*SPRAY|INJ|INJECTION|INFUSION|IV|IM|DEPOT|SACHET|SACHETS|GRANULES|1SAC)(?:\b|_)",
        eval_composite,
        re.IGNORECASE
    )) or (catalog_cat in {
        "SYRUPS", "EYE/EAR DRPOS", "ORAL DROPS", "OINTMENTS/CREAMS/GEL/PASTE", "LOTIONS/SHAMPOOS",
        "SOAPS", "GEL", "SURGICAL", "CONTAINERS", "NASAL SPRAY", "SACHETS", "POWDERS", "GENERAL",
        "RESPULES", "RESPULES/INHALATIONS", "INHALATIONS"
    })

    if is_non_solid and not (is_insulin_device or is_inhaler_device):
        matched_form = "NON_SOLID_FORMULATION"
        if re.search(r"\b(?:SYP|SYRUP|SUSP|SUSPENSION|ELIXIR|LIQ|LIQUID|LINCTUS|EXPECTORANT)\b", eval_composite, re.IGNORECASE) or catalog_cat == "SYRUPS":
            matched_form = "FORMULATION_SYRUPS_AND_SUSPENSIONS"
        elif re.search(r"\b(?:DROP|DROPS|DRPOS|OPHTH|OTIC|EYE\s*DROP|EAR\s*DROP|NASAL\s*DROP)\b", eval_composite, re.IGNORECASE) or catalog_cat in ("EYE/EAR DRPOS", "ORAL DROPS"):
            matched_form = "FORMULATION_DROPS_OPHTHALMIC_OTIC"
        elif re.search(r"\b(?:RESPULE|RESPULES|RESPULS|RESPUELS|RESP|NEB|NEBU|NEBULES|NEBULIZER|NEBULISER|MISTPULES|SMARTULES)\b", eval_composite, re.IGNORECASE) or catalog_cat in ("RESPULES", "RESPULES/INHALATIONS", "INHALATIONS"):
            matched_form = "FORMULATION_RESPULES_AND_NEBULIZERS"
        elif re.search(r"\b(?:OINT|CREAM|CRM|GEL|LOTION|SOAP|SHAMPOO|WASH|PASTE)\b", eval_composite, re.IGNORECASE) or catalog_cat in ("OINTMENTS/CREAMS/GEL/PASTE", "LOTIONS/SHAMPOOS", "SOAPS", "GEL"):
            matched_form = "FORMULATION_TOPICALS_DERMA"
        elif re.search(r"\b(?:SPRAY|NASAL\s*SPRAY)\b", eval_composite, re.IGNORECASE) or catalog_cat == "NASAL SPRAY":
            matched_form = "FORMULATION_SPRAYS"
        elif re.search(r"\b(?:INJ|INJECTION|INFUSION|IV|IM)\b", eval_composite, re.IGNORECASE) or catalog_cat == "INJECTIONS":
            matched_form = "FORMULATION_INJECTIONS"
        elif re.search(r"\b(?:SACHET|SACHETS|GRANULES|1SAC)\b", eval_composite, re.IGNORECASE) or catalog_cat == "SACHETS":
            matched_form = "FORMULATION_SACHETS_GRANULES"

        return {
            "is_chronic_eligible": False,
            "category": matched_form,
            "matched_rule": f"FORMULATION_GATE:{matched_form}",
            "exclusion_reason": f"Non-solid/Acute formulation excluded ({matched_form.replace('_', ' ').title()}).",
        }

    # 3. Hard Baseline Exclusions (Pediatric/Kids & Acute PRN Analgesics)
    if re.search(r"(?:\b|_)(KID|KIDS|KIDZ|PED|PEDIATRIC|PAEDIATRIC)(?:\b|_)", raw_name, re.IGNORECASE):
        return {
            "is_chronic_eligible": False,
            "category": "PEDIATRIC_AND_KIDS_MEDICATIONS",
            "matched_rule": "PEDIATRIC_KID_KEYWORD_MATCH",
            "exclusion_reason": "Pediatric / Kid formulation (Non-chronic).",
        }

    if re.search(r"(?:\b|_)(DOLO|DOLO\s*650|DOLO\s*650MG|DOLO\s*500|CALPOL|PARACETAMOL|MEFTAL|MEFTAL\s*SPAS|DISPRIN|SARIDON|COMBIFLAM)(?:\b|_)", raw_name, re.IGNORECASE):
        return {
            "is_chronic_eligible": False,
            "category": "ACUTE_AND_PRN_ANALGESICS",
            "matched_rule": "ACUTE_ANALGESIC_EXCLUSION",
            "exclusion_reason": "Acute / PRN analgesic / antipyretic (Non-chronic).",
        }

    # 4. Acute / PRN Molecules & Salts Gate (Antibiotics, Analgesics, Antacids, Allergies)
    has_acute_mol = any(m in ACUTE_EXCLUDED_SALT_MOLECULES for m in molecules) or any(m in ACUTE_EXCLUDED_SALT_MOLECULES for m in raw_salt.split())
    if has_acute_mol:
        return {
            "is_chronic_eligible": False,
            "category": "ACUTE_SALT_EXCLUSION",
            "matched_rule": "ACUTE_MOLECULE_MATCH",
            "exclusion_reason": "Contains acute/PRN active molecule or antibiotic/antacid salt.",
        }

    # 5. Check Dynamic Supabase Exclusions
    for p in _DYNAMIC_EXCLUSION_REGEX:
        if is_insulin_device and "INJ" in p.pattern:
            continue
        if p.search(raw_name) or (raw_salt and p.search(raw_salt)):
            return {
                "is_chronic_eligible": False,
                "category": "DYNAMIC_SUPABASE_EXCLUSION",
                "matched_rule": f"SUPABASE_EXCLUSION:{p.pattern}",
                "exclusion_reason": f"Non-chronic item (Excluded by pharmacist rule: '{p.pattern}').",
            }

    # 6. Check for Non-Chronic / OTC / FMCG / Acute exclusions (Code Baseline)
    for category, pattern_list in _EXCLUSION_REGEX_MAP.items():
        for pattern in pattern_list:
            if pattern.search(raw_name) or (raw_salt and pattern.search(raw_salt)):
                if is_insulin_device and category == "FORMULATION_INJECTIONS":
                    continue
                clean_cat = category.replace("_", " ").title()
                return {
                    "is_chronic_eligible": False,
                    "category": category,
                    "matched_rule": f"EXCLUDED_PATTERN:{pattern.pattern}",
                    "exclusion_reason": f"Non-chronic item ({clean_cat}).",
                }

    # 6b. Insulin (pens, cartridges, refills, vials) is chronic maintenance therapy regardless of
    # brand - detected by active salt or device + insulin brand markers, not only by Supabase rules.
    from refillcare.data.packing import is_insulin_product

    if is_insulin_device or is_insulin_product(raw_name, raw_pack, raw_salt):
        return {
            "is_chronic_eligible": True,
            "category": "CHRONIC_MEDICATION",
            "matched_rule": "CHRONIC_INSULIN_THERAPY",
            "exclusion_reason": None,
        }

    # 7. Check Dynamic Supabase Chronic Markers
    for p in _DYNAMIC_CHRONIC_REGEX:
        if p.search(raw_name) or (raw_salt and p.search(raw_salt)):
            return {
                "is_chronic_eligible": True,
                "category": "CHRONIC_MEDICATION",
                "matched_rule": f"SUPABASE_CHRONIC:{p.pattern}",
                "exclusion_reason": None,
            }

    # 8. Check Chronic Chemical Salt Molecules & Brand Markers (Code Baseline)
    has_chronic_mol = any(m in CHRONIC_SALT_MOLECULES for m in molecules)
    is_strong_chronic = any(p.search(raw_name) for p in _CHRONIC_MARKER_REGEX) or (raw_salt and any(p.search(raw_salt) for p in _CHRONIC_MARKER_REGEX))

    # Check packing clues (bulk FMCG sizes)
    if any(p in raw_pack for p in ["1KG", "2KG", "500GM", "400GM", "200GM"]):
        if not (has_chronic_mol or is_strong_chronic):
            return {
                "is_chronic_eligible": False,
                "category": "NUTRITION_FMCG_AND_SACHETS",
                "matched_rule": f"PACKING_BULK_FMCG:{raw_pack}",
                "exclusion_reason": "Bulk FMCG / Nutrition / Sachet pack size.",
            }

    if has_chronic_mol:
        return {
            "is_chronic_eligible": True,
            "category": "CHRONIC_MEDICATION",
            "matched_rule": f"CHRONIC_SALT_MOLECULE_MATCH:{raw_salt}",
            "exclusion_reason": None,
        }

    if is_strong_chronic:
        return {
            "is_chronic_eligible": True,
            "category": "CHRONIC_MEDICATION",
            "matched_rule": "CHRONIC_MARKER_MATCH",
            "exclusion_reason": None,
        }

    # 9. Fail-Safe Default: If item does not positively match any chronic therapeutic class,
    # default to non-chronic (Apollo 24/7 / 1mg standard to prevent acute medication reminder leakage).
    return {
        "is_chronic_eligible": False,
        "category": "UNCLASSIFIED_NON_CHRONIC",
        "matched_rule": "NO_CHRONIC_THERAPY_MATCH",
        "exclusion_reason": "Item does not belong to a recognized chronic maintenance therapeutic class (Fail-safe default).",
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
    salt_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Multi-tiered verification combining Rule-Based Clinical Parsing + Behavioral Consensus.

    Tier 1: Clinical Brand/Salt & Formulation Rules (MedicationClassifier)
    Tier 2: Personal Patient Repurchase Cadence (>= 3 recurring purchases)
    Tier 3: Cross-Customer Population Consensus (Repeat purchase rate >= 35%)
    """
    rule_res = classify_medication_cached(item_name, packing, salt_name)
    
    # Strict Clinical Gate: If explicitly excluded by rules (formulation gate, acute molecules, Supabase exclusion) -> Always False
    if not rule_res["is_chronic_eligible"]:
        if rule_res.get("matched_rule") != "NO_CHRONIC_THERAPY_MATCH":
            return rule_res

    # If personal history demonstrates proven longitudinal adherence for unclassified solid items
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


@lru_cache(maxsize=65536)
def classify_medication_cached(
    item_name: str,
    packing: Optional[str] = None,
    salt_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Memoised ``classify_medication`` (returns a shared dict - do not mutate)."""
    get_dynamic_classification_rules()  # keeps the rule TTL / change-detection ticking
    return classify_medication(item_name, packing=packing, salt_name=salt_name)


@lru_cache(maxsize=32768)
def is_chronic_medication(
    item_name: str,
    packing: Optional[str] = None,
    salt_name: Optional[str] = None,
) -> bool:
    """Convenience boolean helper to check if an item is eligible chronic medication."""
    res = classify_medication(item_name, packing=packing, salt_name=salt_name)
    return bool(res["is_chronic_eligible"])


def filter_chronic_transactions(
    df: pd.DataFrame,
    item_col: str = "itemName",
    packing_col: str = "packing",
    salt_col: Optional[str] = "SALT",
) -> pd.DataFrame:
    """Filter a DataFrame of transactions to keep only eligible chronic medication records.

    Args:
        df: Input transactions DataFrame.
        item_col: Name of the item name column.
        packing_col: Name of the packing column if available.
        salt_col: Name of the salt composition column if available.

    Returns:
        Filtered DataFrame containing only chronic medication purchases.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if item_col not in df.columns:
        return df.copy()

    items = df[item_col].fillna("").astype(str).tolist()
    packings = df[packing_col].fillna("").astype(str).tolist() if packing_col in df.columns else [None] * len(df)
    salts = df[salt_col].fillna("").astype(str).tolist() if (salt_col and salt_col in df.columns) else [None] * len(df)

    eligible_mask = [
        is_chronic_medication(item, pack, salt)
        for item, pack, salt in zip(items, packings, salts)
    ]

    return pd.DataFrame(df[pd.Series(eligible_mask, index=df.index)].copy())


