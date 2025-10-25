import json, os
from PIL import Image
meta_in = "data/data/preprocessed_data/val_meta_fixedpaths_with_radar.json"
meta_out = "data/data/preprocessed_data/val_meta_fixedpaths_with_radar_labelled_interactive.json"
N = 100   # label first N images interactively
m = json.load(open(meta_in))
count = 0
for i, it in enumerate(m):
    if i >= N: break
    p = it.get("processed_path") or it.get("original_path")
    if not p or not os.path.exists(p):
        continue
    print(f"Index {i}: {p}")
    img = Image.open(p); img.show()
    lbl = input("Enter integer label (blank to skip): ").strip()
    try:
        if lbl != '':
            it['label'] = int(lbl); count+=1
    except:
        print("Invalid; skipping")
    img.close()
print("Attached", count, "labels. Writing", meta_out)
json.dump(m, open(meta_out,"w"), indent=2)