"""
Production-grade CSV Column Alias and Canonical Field Normalization System.

Supports flexible client CSV formats for the Mediastra Image + Text Campaign.
Handles:
- Header normalization (case, whitespace, punctuation, hyphens, underscores, slashes, UTF-8 BOM)
- Multi-alias resolution to canonical fields
- Ambiguity detection (preventing silent wrong column selection)
- Missing required column detection with accepted alias examples
- Unmapped/extra column detection (ignoring extra columns safely)
- Duplicate column detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical Fields Definition
# ---------------------------------------------------------------------------
CANONICAL_CUSTOMER_NAME = "customer_name"
CANONICAL_PHONE = "phone"
CANONICAL_MEDICINE = "medicine"
CANONICAL_BRANCH = "branch"
CANONICAL_STORE_NAME = "store_name"
CANONICAL_CONTACT_NO = "contact_no"
CANONICAL_MANAGER_CONTACT = "manager_contact"

ALL_CANONICAL_FIELDS: List[str] = [
    CANONICAL_CUSTOMER_NAME,
    CANONICAL_PHONE,
    CANONICAL_MEDICINE,
    CANONICAL_BRANCH,
    CANONICAL_STORE_NAME,
    CANONICAL_CONTACT_NO,
    CANONICAL_MANAGER_CONTACT,
]

REQUIRED_CANONICAL_FIELDS: List[str] = [
    CANONICAL_CUSTOMER_NAME,
    CANONICAL_PHONE,
    CANONICAL_MEDICINE,
    CANONICAL_BRANCH,
]

OPTIONAL_CANONICAL_FIELDS: List[str] = [
    CANONICAL_STORE_NAME,
    CANONICAL_CONTACT_NO,
    CANONICAL_MANAGER_CONTACT,
]

# Internal standard DataFrame column names downstream expects
CANONICAL_TO_INTERNAL_MAP: Dict[str, str] = {
    CANONICAL_CUSTOMER_NAME: "Name",
    CANONICAL_PHONE: "Phone number",
    CANONICAL_MEDICINE: "Medicine",
    CANONICAL_BRANCH: "Branch",
    CANONICAL_STORE_NAME: "Store Name",
    CANONICAL_CONTACT_NO: "Contact No.",
    CANONICAL_MANAGER_CONTACT: "Manager Contact",
}

INTERNAL_TO_CANONICAL_MAP: Dict[str, str] = {
    v: k for k, v in CANONICAL_TO_INTERNAL_MAP.items()
}


# ---------------------------------------------------------------------------
# Header Normalization
# ---------------------------------------------------------------------------
def normalize_header_string(header: str) -> str:
    """
    Normalize raw header strings for flexible comparison.

    - Strips UTF-8 BOM (\\ufeff) and leading/trailing whitespace
    - Lowercases
    - Converts underscores, hyphens, slashes, backslashes, dots, parentheses, colons to spaces
    - Collapses multiple whitespace characters to a single space
    """
    if header is None:
        return ""
    text = str(header).lstrip("\ufeff").strip().lower()
    # Replace punctuation / delimiters with space
    text = re.sub(r"[_\-\/\\\.\(\):,]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Central Alias Configuration
# ---------------------------------------------------------------------------
COLUMN_ALIASES: Dict[str, List[str]] = {
    CANONICAL_CUSTOMER_NAME: [
        "customer name",
        "customer_name",
        "name",
        "full name",
        "full_name",
        "fullname",
        "customer",
        "patient name",
        "patient_name",
        "patient",
        "client name",
        "client_name",
        "client",
        "recipient name",
        "recipient_name",
        "recipient",
        "user name",
        "username",
        "cust name",
    ],
    CANONICAL_PHONE: [
        "phone",
        "phone number",
        "phone_number",
        "phone no",
        "phone no.",
        "phone_no",
        "phonenumber",
        "phoneno",
        "mobile",
        "mobile number",
        "mobile_number",
        "mobile no",
        "mobile no.",
        "mobile_no",
        "mobilenumber",
        "mobileno",
        "whatsapp",
        "whatsapp number",
        "whatsapp_number",
        "whatsapp no",
        "whatsapp no.",
        "whatsapp_no",
        "whatsapp mobile",
        "whatsapp phone",
        "whatsappphone",
        "whatsappmobile",
        "customer phone",
        "customer mobile",
        "customer whatsapp",
        "recipient phone",
        "recipient mobile",
        "recipient whatsapp",
        "patient phone",
        "patient mobile",
        "client phone",
        "client mobile",
    ],
    CANONICAL_MEDICINE: [
        "medicine",
        "medicine name",
        "medicine_name",
        "medicines",
        "medication",
        "medication name",
        "medication_name",
        "medications",
        "med",
        "meds",
        "drug",
        "drug name",
        "drug_name",
        "drugs",
        "product",
        "product name",
        "product_name",
        "products",
        "item",
        "item name",
        "medicine list",
        "medication list",
        "customer medication list",
        "customer_medication_list",
        "customer medicine list",
        "customer_medicine_list",
        "refill medicine",
        "refill medicines",
        "refill medication",
        "refill medications",
        "prescribed medicine",
        "prescribed medicines",
        "prescription",
        "rx",
    ],
    CANONICAL_BRANCH: [
        "branch",
        "branch name",
        "branch_name",
        "branch location",
        "branch_location",
        "branch / location",
        "location",
        "location name",
        "outlet",
        "outlet name",
        "outlet_name",
        "store branch",
        "store_branch",
        "store location",
        "store_location",
        "pharmacy branch",
        "pharmacy location",
        "center",
        "centre",
    ],
    CANONICAL_STORE_NAME: [
        "store",
        "store name",
        "store_name",
        "pharmacy",
        "pharmacy name",
        "pharmacy_name",
        "shop",
        "shop name",
        "shop_name",
        "business name",
        "business_name",
        "organization",
        "organization name",
        "company",
        "company name",
    ],
    CANONICAL_CONTACT_NO: [
        "store contact",
        "store contact no",
        "store contact no.",
        "store contact number",
        "store_contact",
        "store phone",
        "store_phone",
        "store mobile",
        "store mobile number",
        "store_mobile",
        "pharmacy phone",
        "pharmacy contact",
        "pharmacy contact no",
        "pharmacy contact number",
        "pharmacy_contact",
        "pharmacy_phone",
        "contact no",
        "contact no.",
        "contact_no",
        "contact number",
        "contact_number",
        "store phone number",
        "pharmacy phone number",
        "helpline",
        "store helpline",
        "branch contact",
    ],
    CANONICAL_MANAGER_CONTACT: [
        "manager contact",
        "manager contact no",
        "manager contact no.",
        "manager contact number",
        "manager_contact",
        "manager phone",
        "manager_phone",
        "manager mobile",
        "manager mobile number",
        "manager_mobile",
        "manager number",
        "manager_number",
        "manager no",
        "manager no.",
        "manager_no",
        "manager phone number",
        "supervisor contact",
        "supervisor phone",
    ],
}

# Precompute normalized alias lookup set for fast, normalized matching
_NORMALIZED_ALIAS_LOOKUP: Dict[str, Set[str]] = {
    canonical: {normalize_header_string(a) for a in aliases}
    for canonical, aliases in COLUMN_ALIASES.items()
}


# ---------------------------------------------------------------------------
# Resolution Result Data Structure
# ---------------------------------------------------------------------------
@dataclass
class AliasResolutionResult:
    is_valid: bool
    rename_map: Dict[str, str] = field(default_factory=dict)
    detected_mappings: Dict[str, str] = field(default_factory=dict)
    canonical_to_raw: Dict[str, str] = field(default_factory=dict)
    unmapped_columns: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    ambiguities: Dict[str, List[str]] = field(default_factory=dict)
    duplicate_columns: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_columns: List[str] = field(default_factory=list)

    def summary_text(self) -> str:
        """User-friendly summary of the alias resolution."""
        lines = []
        if self.detected_mappings:
            lines.append("Detected column mappings:")
            for raw_col, canonical in self.detected_mappings.items():
                internal_name = CANONICAL_TO_INTERNAL_MAP.get(canonical, canonical)
                lines.append(f"  • '{raw_col}' → {canonical} ({internal_name})")
        if self.unmapped_columns:
            lines.append("\nUnmapped/Extra columns (ignored):")
            for col in self.unmapped_columns:
                lines.append(f"  • '{col}'")
        if self.missing_required:
            lines.append("\nMissing required canonical fields:")
            for field_name in self.missing_required:
                examples = ", ".join(COLUMN_ALIASES.get(field_name, [])[:4])
                lines.append(f"  • {field_name} (Examples: {examples})")
        if self.ambiguities:
            lines.append("\nAmbiguous column mappings:")
            for field_name, cols in self.ambiguities.items():
                lines.append(f"  • {field_name}: multiple matching columns {cols}")
        if self.duplicate_columns:
            lines.append("\nDuplicate column names:")
            for col in self.duplicate_columns:
                lines.append(f"  • '{col}'")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resolution Function
# ---------------------------------------------------------------------------
def resolve_column_aliases(columns: List[str]) -> AliasResolutionResult:
    """
    Resolve raw CSV column headers into canonical internal fields.

    Parameters:
        columns: List of raw column names from uploaded CSV/XLSX.

    Returns:
        AliasResolutionResult with mappings, unmapped columns, ambiguities,
        missing fields, and validation status.
    """
    raw_cols = [c for c in columns if c is not None and str(c).strip() and not str(c).startswith("Unnamed:")]
    
    # 1. Check for duplicate column names
    seen_raw: Set[str] = set()
    duplicates: List[str] = []
    for c in raw_cols:
        norm = normalize_header_string(c)
        if norm in seen_raw:
            duplicates.append(c)
        seen_raw.add(norm)

    rename_map: Dict[str, str] = {}
    detected_mappings: Dict[str, str] = {}
    canonical_to_raw: Dict[str, str] = {}
    ambiguities: Dict[str, List[str]] = {}
    mapped_raw_cols: Set[str] = set()
    errors: List[str] = []

    if duplicates:
        errors.append(f"Duplicate column names detected: {', '.join(duplicates)}. Please ensure each column is unique.")

    # 2. First Pass: Exact matches for canonical / internal names
    for canonical in ALL_CANONICAL_FIELDS:
        internal_name = CANONICAL_TO_INTERNAL_MAP[canonical]
        norm_canonical = normalize_header_string(canonical)
        norm_internal = normalize_header_string(internal_name)

        exact_matches = [
            c for c in raw_cols
            if c not in mapped_raw_cols and (
                normalize_header_string(c) == norm_canonical or
                normalize_header_string(c) == norm_internal
            )
        ]

        if len(exact_matches) == 1:
            raw_col = exact_matches[0]
            canonical_to_raw[canonical] = raw_col
            detected_mappings[raw_col] = canonical
            rename_map[raw_col] = internal_name
            mapped_raw_cols.add(raw_col)
        elif len(exact_matches) > 1:
            ambiguities[canonical] = exact_matches

    # 3. Second Pass: Alias matches for remaining unmapped canonical fields
    for canonical in ALL_CANONICAL_FIELDS:
        if canonical in canonical_to_raw or canonical in ambiguities:
            continue

        alias_set = _NORMALIZED_ALIAS_LOOKUP.get(canonical, set())
        matching_cols = [
            c for c in raw_cols
            if c not in mapped_raw_cols and normalize_header_string(c) in alias_set
        ]

        if len(matching_cols) == 1:
            raw_col = matching_cols[0]
            canonical_to_raw[canonical] = raw_col
            detected_mappings[raw_col] = canonical
            rename_map[raw_col] = CANONICAL_TO_INTERNAL_MAP[canonical]
            mapped_raw_cols.add(raw_col)
        elif len(matching_cols) > 1:
            ambiguities[canonical] = matching_cols

    # 4. Check for missing required canonical fields
    missing_required: List[str] = []
    for req in REQUIRED_CANONICAL_FIELDS:
        if req not in canonical_to_raw:
            missing_required.append(req)

    # 5. Format error messages
    if ambiguities:
        for canon, cols in ambiguities.items():
            errors.append(
                f"Ambiguous column mapping for canonical field '{canon}': "
                f"multiple columns found {cols}. Please retain only one or rename."
            )

    if missing_required:
        for req in missing_required:
            examples = ", ".join(COLUMN_ALIASES.get(req, [])[:4])
            errors.append(
                f"Missing required field '{req}'. "
                f"Accepted alias examples include: [{examples}]. "
                f"Actual columns received: {raw_cols}"
            )

    # 6. Remaining columns are unmapped
    unmapped_columns = [c for c in raw_cols if c not in mapped_raw_cols]

    is_valid = (
        len(errors) == 0 and
        len(missing_required) == 0 and
        len(ambiguities) == 0 and
        len(duplicates) == 0
    )

    return AliasResolutionResult(
        is_valid=is_valid,
        rename_map=rename_map,
        detected_mappings=detected_mappings,
        canonical_to_raw=canonical_to_raw,
        unmapped_columns=unmapped_columns,
        missing_required=missing_required,
        ambiguities=ambiguities,
        duplicate_columns=duplicates,
        errors=errors,
        raw_columns=raw_cols,
    )


# ---------------------------------------------------------------------------
# DataFrame Normalizer using the Alias System
# ---------------------------------------------------------------------------
def normalize_dataframe_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, AliasResolutionResult]:
    """
    Apply alias resolution to a pandas DataFrame.

    Renames columns to internal standard names ('Name', 'Phone number', etc.)
    and returns (normalized_df, resolution_result).
    """
    if df is None:
        return df, AliasResolutionResult(is_valid=False, errors=["DataFrame is None"])

    result = resolve_column_aliases(list(df.columns))
    if not result.is_valid:
        return df, result

    # Rename mapped columns
    renamed_df = df.rename(columns=result.rename_map)
    return renamed_df, result
