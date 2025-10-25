import os, json, re
from glob import glob

meta_in  = "data/data/preprocessed_data/val_meta_fixedpaths_with_radar.json"
meta_out = "data/data/preprocessed_data/val_meta_fixedpaths_with_radar_with_dacc.json"
dacc_root = "data/data/preprocessed_data/dacc"
time_tol_ms = 5000  # tolerance (ms) for nearest match - increase if needed

def parse_ts(fn):
    groups = re.findall(r'(\d{7,})', os.path.basename(fn))
    if not groups: return None
    return int(groups[-1])

# index all dacc files
dacc_paths = [p for p in glob(os.path.join(dacc_root, "**/*.*"), recursive=True)
              if p.lower().endswith(('.npz', '.npy')) and os.path.isfile(p)]
print("Indexed dacc files:", len(dacc_paths))

index = []
for p in dacc_paths:
    bn = os.path.basename(p)
    ts = parse_ts(bn)
    cam_match = re.search(r'(CAM_[A-Z_]+)', bn.upper())
    cam = cam_match.group(1) if cam_match else None
    index.append({"path": p, "bn": bn, "ts": ts, "cam": cam})

ts_map = {}
for it in index:
    if it["ts"] is not None:
        ts_map.setdefault(it["ts"], []).append(it)

def find_by_cam_and_ts(cam, ts):
    if ts is None: return None
    lst = ts_map.get(ts, [])
    for it in lst:
        if it["cam"] == cam:
            return it["path"]
    return None

def find_by_ts_any(ts):
    if ts is None: return None
    lst = ts_map.get(ts, [])
    return lst[0]["path"] if lst else None

def find_nearest(cam, ts, tol=time_tol_ms):
    if ts is None: return None
    best = None; bestscore = None
    for it in index:
        if it["ts"] is None: continue
        diff = abs(it["ts"] - ts)
        if diff > tol: continue
        score = diff * (0.5 if it["cam"]==cam and cam is not None else 1.0)
        if best is None or score < bestscore:
            best = it; bestscore = score
    return best["path"] if best else None

def find_by_last_digits(ts, ndigits=9):
    if ts is None: return None
    s = str(ts)[-ndigits:]
    for it in index:
        if s in it["bn"]:
            return it["path"]
    return None

meta = json.load(open(meta_in))
matched = 0
for it in meta:
    cam = it.get("camera")
    ts = it.get("timestamp_ms")
    if isinstance(ts, str) and ts.isdigit(): ts = int(ts)
    p = find_by_cam_and_ts(cam, ts)
    if p:
        it["dacc_path"] = p; matched += 1; continue
    p = find_by_ts_any(ts)
    if p:
        it["dacc_path"] = p; matched += 1; continue
    p = find_nearest(cam, ts)
    if p:
        it["dacc_path"] = p; matched += 1; continue
    p = find_by_last_digits(ts)
    if p:
        it["dacc_path"] = p; matched += 1; continue
    it["dacc_path"] = None

print("Matched", matched, "of", len(meta))
json.dump(meta, open(meta_out, "w"), indent=2)
print("Wrote:", meta_out)