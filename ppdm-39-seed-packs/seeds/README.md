# PPDM 3.9 Extended Seed Pack (Reference Tables)

Generated: 2026-01-09

## What this is
A larger starter set of **reference (r_ / ra_) seed JSON files** in the exact format your **Seed R Tables** page expects:

```json
{
  "name": "dbo.r_dir_srvy_type",
  "model": "ppdm39",
  "version": "1.0",
  "rows": [
    {"DIR_SRVY_TYPE":"MWD","LONG_NAME":"Measurement While Drilling","SOURCE":"SYNTH","ACTIVE_IND":"Y"}
  ]
}
```

## Where the files are
`seeds/ppdm39/r_extended/*.json`

## How to use in the app
1. Open **Seed R Tables** page.
2. Pick the target table (e.g. `dbo.r_well_status`).
3. Upload the matching JSON file from this pack.
4. Click **Use seed JSON for this table**.
5. In the mapping grid:
   - Map the PK column(s) at minimum.
   - Map any required NOT NULL columns (varies by schema).
   - Set constants if needed (SOURCE='SYNTH', ACTIVE_IND='Y', etc).
6. **Compute missing** → **Seed missing now**.

## Important notes
- PPDM implementations differ. If your DB uses a different PK column name, keep the JSON as your *source* and map it correctly in Step 2.
- If a given r_ table doesn't exist in your DB, skip it.
- This pack is intended to get you unblocked fast; expand/replace with your organization’s authoritative code lists later.