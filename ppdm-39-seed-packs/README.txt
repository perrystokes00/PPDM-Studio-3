PPDM 3.9 — Starter R-Seed Pack (JSON)
====================================

This zip contains a practical starter pack of common PPDM 3.9 r_* reference tables
used in well header + status + directional survey workflows.

Each file is compatible with your Seed R Tables page (Option A):
  {
    "name": "dbo.r_table_name",
    "rows": [ { ... }, ... ]
  }

How to use:
1) Open: Seed R Tables
2) Select target table (e.g., dbo.r_well_status)
3) Upload the matching JSON file from: seeds/ppdm39/r/
4) Click "Use seed JSON for this table"
5) Compute missing -> Seed missing

Included:
r_source
r_ppdm_row_quality
r_confidential_type
r_plot_symbol
r_well_status_type
r_well_status
r_well_class
r_well_datum_type
r_well_level_type
r_well_profile_type
r_dir_srvy_type
r_srvy_type
