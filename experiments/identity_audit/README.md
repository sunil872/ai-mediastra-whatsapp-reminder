# Customer-Item Identity Impact Audit Module

This directory contains the non-destructive audit artifacts evaluating the composite `customer_item_key`:
`customer_item_key = normalized_store_id + normalized_phone + normalized_customer_name + normalized_item_id`

## Generated Audit Artifacts

- `identity_normalization.py`: Conservative normalizers for store, phone, customer name, and item ID.
- `run_identity_audit.py`: Audit pipeline comparing current vs proposed entities.
- `identity_audit_report.json`: Machine-readable summary statistics.
- `identity_audit_report.md`: Human-readable impact analysis and breakdown.
- `identity_entity_comparison.csv`: Entity-by-entity comparison of purchase counts, stability, and eligibility.
- `identity_split_cases.csv`: Detailed cases where 1 current entity splits into multiple proposed keys.
- `identity_merge_cases.csv`: Detailed cases where multiple current customer IDs merge into 1 proposed key.
- `path_a_impact.csv`: Slice of entities whose Path A eligibility status changes.
- `path_b_impact.csv`: Slice of entities whose Path B eligibility status changes.
