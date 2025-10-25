#!/usr/bin/env python3
"""
Attach nearest radar (.npy) and dacc (.npz) files to metadata entries by timestamp.
This version uses recursive globbing and is robust to files either at the root or in camera subfolders.
"""
import json, os, argparse, glob
from bisect import bisect_left
import re

def parse_ts(fn):
    # find last long digit sequence (>=8 digits); return int or None
    m = re.findall(r'(\d{8,})', fn)
    if not m:
        return None
    return int(m[-1])

def build_index(folder_root, pattern="**/*.*"):
    # use recursive glob
    glob_pattern = os.path.join(folder_root, pattern)
    files = sorted(glob.glob(glob_pattern, recursive=True))
    idx = []
    for f in files:
        if not os.path.isfile(f):
            continue
        ts = parse_ts(os.path.basename(f))
        if ts is not None:
            idx.append((ts, f))
    idx.sort()
    ts_list = [t for t,_ in idx]
    files_list = [p for _,p in idx]
    return ts_list, files_list

def find_nearest(ts, ts_list, files_list):
    if not ts_list:
        return None
    i = bisect_left(ts_list, ts)
    cand = []
    if i < len(ts_list):
        cand.append((abs(ts_list[i]-ts), files_list[i]))
    if i-1 >= 0:
        cand.append((abs(ts_list[i-1]-ts), files_list[i-1]))
    cand.sort()
    return cand[0][1] if cand else None

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--radar-root", default="data/data/preprocessed_data/radar")
    p.add_argument("--dacc-root",  default="data/data/preprocessed_data/dacc")
    p.add_argument("--radar-pattern", default="**/*.npy")
    p.add_argument("--dacc-pattern",  default="**/*.npz")
    args = p.parse_args()

    print("Indexing radar files (recursive)...")
    radar_ts, radar_files = build_index(args.radar_root, args.radar_pattern)
    print("Found radar files:", len(radar_files))
    print("Indexing dacc files (recursive)...")
    dacc_ts, dacc_files = build_index(args.dacc_root, args.dacc_pattern)
    print("Found dacc files:", len(dacc_files))

    meta = json.load(open(args.meta))
    out = []
    missed_radar = missed_dacc = 0
    for it in meta:
        ts = it.get("timestamp_ms")
        if ts is None:
            it["radar_path"] = None
            it["dacc_path"] = None
            out.append(it)
            continue
        r = find_nearest(ts, radar_ts, radar_files)
        d = find_nearest(ts, dacc_ts, dacc_files)
        if r is None:
            missed_radar += 1
        if d is None:
            missed_dacc += 1
        it["radar_path"] = r
        it["dacc_path"] = d
        out.append(it)

    json.dump(out, open(args.out, "w"), indent=2)
    print("Wrote", args.out, "missed_radar:", missed_radar, "missed_dacc:", missed_dacc)
