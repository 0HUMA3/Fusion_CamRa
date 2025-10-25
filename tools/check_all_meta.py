#!/usr/bin/env python3
import json, os, argparse

META_FILES = [
    "data/data/preprocessed_data/train_meta.json",
    "data/data/preprocessed_data/val_meta.json",
    "data/data/preprocessed_data/test_meta.json",
    "data/data/preprocessed_data/train_meta_fixedpaths.json",
    "data/data/preprocessed_data/val_meta_fixedpaths.json",
    "data/data/preprocessed_data/test_meta_fixedpaths.json",
]

def check(meta_path):
    if not os.path.exists(meta_path):
        print(f"Missing meta: {meta_path}")
        return
    with open(meta_path) as f:
        data = json.load(f)
    total = len(data)
    has_processed = sum(1 for it in data if it.get("processed_path"))
    has_original  = sum(1 for it in data if it.get("original_path"))
    missing_proc = sum(1 for it in data if not it.get("processed_path") or not os.path.exists(it.get("processed_path","")))
    missing_orig = sum(1 for it in data if not it.get("original_path") or not os.path.exists(it.get("original_path","")))
    example = data[:2]
    print("----")
    print(meta_path)
    print(" total entries:", total)
    print(" entries with processed_path:", has_processed)
    print(" entries with original_path:", has_original)
    print(" missing processed_path on disk:", missing_proc)
    print(" missing original_path on disk:", missing_orig)
    print(" sample entries:", example)

if __name__ == "__main__":
    import sys
    files = sys.argv[1:] or META_FILES
    for m in files:
        check(m)
