import json
from pathlib import Path

PDF_DIR = Path(r"docs/schema_pdfs")

seed_path = PDF_DIR / "bookmarks_seed_ppdm39_toc.json"
auto_path = PDF_DIR / "bookmarks.json"
out_path  = PDF_DIR / "bookmarks.json"   # overwrite in place

def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

seed = load_json(seed_path)   # your rich PPDM 3.9 TOC
auto = load_json(auto_path)   # your extracted bookmarks (Lite is good; PPDM39 is junk)

merged = {}

# Start with auto (so Lite comes through)
for pdf_name, bm in auto.items():
    if isinstance(bm, dict):
        merged[pdf_name] = dict(bm)

# Overlay seed (so PPDM39 TOC replaces junk)
for pdf_name, bm in seed.items():
    if not isinstance(bm, dict):
        continue
    merged.setdefault(pdf_name, {})
    # overwrite/merge keys
    merged[pdf_name].update(bm)

# Cleanup: remove "Page:" style junk entries
for pdf_name, bm in list(merged.items()):
    if not isinstance(bm, dict):
        continue
    bad_keys = [k for k in bm.keys() if k.strip().lower() in ("page:", "page")]
    for k in bad_keys:
        bm.pop(k, None)

# Write final
out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

print(f"Wrote merged bookmarks to: {out_path.resolve()}")
for k in merged.keys():
    print(f"{k}: {len(merged[k])} bookmarks")
