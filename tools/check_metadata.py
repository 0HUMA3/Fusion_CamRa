# tools/check_metadata.py
import json, glob, os, sys
meta_files = glob.glob("data/data/*_meta.json")
if not meta_files:
    print("No meta json files found in data/data")
    sys.exit(1)

for mf in meta_files:
    with open(mf) as f:
        data = json.load(f)
    n_items = len(data) if isinstance(data, list) else (len(data.get('images', [])) if isinstance(data, dict) else 0)
    print(os.path.basename(mf), "-> entries:", n_items)
