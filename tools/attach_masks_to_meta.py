# tools/attach_masks_to_meta.py
import os, json, glob

ROOT = "data/data"
MASK_DIR = os.path.join(ROOT, "derived_masks")
META_DIR = os.path.join(ROOT, "preprocessed_data")

meta_files = [
    os.path.join(META_DIR, "train_meta_fixedpaths_with_radar.json"),
    os.path.join(META_DIR, "val_meta_fixedpaths_with_radar.json"),
    os.path.join(META_DIR, "test_meta_fixedpaths_with_radar.json"),
]

def main():
    # build map: basename -> maskpath
    masks = {}
    for f in glob.glob(os.path.join(MASK_DIR, "*.npy")):
        bn = os.path.splitext(os.path.basename(f))[0]
        masks[bn] = os.path.relpath(f, start=os.getcwd())

    for mf in meta_files:
        if not os.path.exists(mf):
            print("Skip:", mf); continue
        with open(mf) as fh:
            data = json.load(fh)
        updated = False
        for item in data:
            # find basename from processed_path (or original_path)
            proc = item.get("processed_path") or item.get("original_path") or ""
            bn = os.path.splitext(os.path.basename(proc))[0]
            mask_rel = masks.get(bn)
            if mask_rel:
                item["mask_path"] = mask_rel
                updated = True
            else:
                item["mask_path"] = None
        out = mf.replace(".json", "_with_masks.json")
        with open(out, "w") as fh:
            json.dump(data, fh, indent=2)
        print("Wrote:", out, "entries:", len(data))

if __name__ == "__main__":
    main()
