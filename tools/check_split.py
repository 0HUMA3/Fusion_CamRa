#!/usr/bin/env python3
import os
import json
import argparse
from collections import Counter

def count_files(folder, exts=(".png",".jpg",".npy",".npz",".pcd",".pcd.bin")):
    c = 0
    for root,_,files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                c += 1
    return c

def from_meta(meta_path):
    with open(meta_path) as f:
        data = json.load(f)
    # Expect list of dicts with a 'split' or path fields
    splits = Counter()
    for it in data:
        sp = it.get("split") or it.get("set") or "train"
        splits[sp] += 1
    return splits, data

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/data", help="root containing original_data / preprocessed_data")
    p.add_argument("--meta", default="", help="optional metadata json to inspect")
    args = p.parse_args()

    print("Filesystem counts (example folders):")
    print("CAM_FRONT_LEFT:", count_files(os.path.join(args.data_root,"original_data","CAM_FRONT_LEFT")))
    print("LIDAR_TOP:", count_files(os.path.join(args.data_root,"original_data","LIDAR_TOP")))
    print("preprocessed dacc:", count_files(os.path.join(args.data_root,"preprocessed_data","dacc")))

    if args.meta:
        splits, data = from_meta(args.meta)
        print("Meta split counts:", dict(splits))
        # sample entries
        print("Example meta entries (first 5):")
        for it in data[:5]:
            print(it)
