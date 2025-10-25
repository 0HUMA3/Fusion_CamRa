# tools/add_radar_to_meta.py
import os, json, glob, re
from bisect import bisect_left

ROOT = "data/data"
RADAR_PREP_DIR = os.path.join(ROOT, "preprocessed_data", "radar")
META_DIR = os.path.join(ROOT, "preprocessed_data")
META_FILES = {
    "train": os.path.join(META_DIR, "train_meta_fixedpaths.json"),
    "val":   os.path.join(META_DIR, "val_meta_fixedpaths.json"),
    "test":  os.path.join(META_DIR, "test_meta_fixedpaths.json"),
}

def extract_timestamp_from_filename(fname):
    m = re.search(r'(\d{12,19})', fname)
    return int(m.group(1)) if m else None

# Build radar index
radar_index = {}
if os.path.isdir(RADAR_PREP_DIR):
    for radar_dir in sorted(os.listdir(RADAR_PREP_DIR)):
        dpath = os.path.join(RADAR_PREP_DIR, radar_dir)
        if not os.path.isdir(dpath): continue
        files = sorted(glob.glob(os.path.join(dpath, "*.npy")))
        ts_list = []
        for f in files:
            ts = extract_timestamp_from_filename(os.path.basename(f))
            if ts is not None:
                ts_list.append((ts, f))
        ts_list.sort()
        radar_index[radar_dir] = ts_list
else:
    print("Radar preprocessed dir not found:", RADAR_PREP_DIR)

def find_closest_radar(radar_list, target_ts):
    if not radar_list: return None
    timestamps = [t for t,_ in radar_list]
    i = bisect_left(timestamps, target_ts)
    candidates = []
    if i>0: candidates.append(radar_list[i-1])
    if i < len(radar_list): candidates.append(radar_list[i])
    best = min(candidates, key=lambda x: abs(x[0]-target_ts))
    return best[1]

for split, meta_path in META_FILES.items():
    if not os.path.exists(meta_path):
        print("Skipping, not found:", meta_path); continue
    with open(meta_path) as f:
        data = json.load(f)
    updated = []
    for item in data:
        ts = item.get("timestamp_ms")
        if ts is None:
            ts = extract_timestamp_from_filename(item.get("processed_path","") or item.get("original_path","") or "")
        radar_path = None
        cam = (item.get("camera") or "").upper()
        radar_dir = None
        if "FRONT_LEFT" in cam: radar_dir = "RADAR_FRONT_LEFT"
        elif "FRONT_RIGHT" in cam: radar_dir = "RADAR_FRONT_RIGHT"
        elif "BACK_LEFT" in cam: radar_dir = "RADAR_BACK_LEFT"
        elif "BACK_RIGHT" in cam: radar_dir = "RADAR_BACK_RIGHT"
        if radar_dir and radar_dir in radar_index and ts:
            radar_path = find_closest_radar(radar_index[radar_dir], ts)
        if radar_path is None and ts:
            best = (None, None)
            for rd,lst in radar_index.items():
                p = find_closest_radar(lst, ts)
                if p:
                    cand_ts = extract_timestamp_from_filename(os.path.basename(p))
                    diff = abs(cand_ts - ts)
                    if best[0] is None or diff < best[0]:
                        best = (diff, p)
            radar_path = best[1]
        item["radar_data"] = os.path.relpath(radar_path, start=os.getcwd()) if radar_path else None
        updated.append(item)
    out = meta_path.replace(".json", "_with_radar.json")
    with open(out, "w") as f:
        json.dump(updated, f, indent=2)
    print("Wrote:", out, "entries:", len(updated))
