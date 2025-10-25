# tools/fix_meta_paths.py
import json, os, re

ROOT = "data/data"
CAM_PREP = os.path.join(ROOT, "preprocessed_data")
META_FILES = [
  os.path.join(ROOT, "preprocessed_data", "camera_metadata.json"),
  os.path.join(ROOT, "preprocessed_data", "train_meta.json"),
  os.path.join(ROOT, "preprocessed_data", "val_meta.json"),
  os.path.join(ROOT, "preprocessed_data", "test_meta.json"),
]

def try_fix_path(p):
    if not p: return p
    p = p.replace("\\","/")
    # Replace common repo prefix with your preprocessed camera folder
    p = re.sub(r"data\/nuscenes(_raw)?\/.*?\/preprocessed_data\/camera", CAM_PREP, p)
    p = p.replace("data/nuscenes", "data/MINOR_PROJECT")
    return p

for mf in META_FILES:
    if not os.path.exists(mf):
        print("Skip missing:", mf); continue
    with open(mf) as f:
        data = json.load(f)
    changed = False
    for item in data:
        for key in ("processed_path","original_path","image","camera_image"):
            if key in item and item[key]:
                newp = try_fix_path(item[key])
                if newp != item[key]:
                    item[key] = newp
                    changed = True
    if changed:
        out = mf.replace(".json", "_fixedpaths.json")
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print("Wrote fixed file:", out)
    else:
        print("No changes for", mf)
