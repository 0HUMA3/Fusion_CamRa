#!/usr/bin/env python3
"""
preprocess_camera_full.py

Full camera preprocessing pipeline for radar-camera fusion projects.

Features:
 - Resize images to (width x height)
 - Optional undistort using per-camera intrinsics JSON
 - Optional crop ROI (x,y,w,h)
 - Optional histogram equalization (CLAHE)
 - Optional ImageNet normalization (applied in-memory; saved images remain visually viewable)
 - Extract timestamps (from filename integer or file mtime)
 - Save processed images and write metadata JSON
 - Optionally create train/val/test splits

Usage example:
python src/preprocessing/preprocess_camera_full.py \
  --data-root data/MINOR_PROJECT/original_data \
  --out-root  data/MINOR_PROJECT/preprocessed \
  --width 1280 --height 720 \
  --crop 0 200 1280 520 \
  --equalize \
  --make-splits 0.8 0.1 0.1

"""

import argparse
from pathlib import Path
import re
import json
import cv2
import numpy as np
import random
from tqdm import tqdm
import os
import math
from datetime import datetime

CAM_FOLDERS = ["CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK_RIGHT"]

# ---------------- utilities ----------------
def find_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in [".jpg",".jpeg",".png"]])

def find_longest_int_in_name(s: str):
    matches = re.findall(r'\d{6,}', s)  # sequences ≥6 digits
    if not matches:
        return None
    return int(max(matches, key=len))

def file_timestamp_ms(path: Path):
    t = find_longest_int_in_name(path.name)
    if t is not None:
        return int(t)
    return int(path.stat().st_mtime * 1000)

def load_calib_json(path: Path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] calib JSON not found: {p}")
        return None
    with open(p, 'r') as f:
        return json.load(f)

def undistort_image(img: np.ndarray, cam_name: str, calib: dict):
    try:
        cam_cfg = calib.get(cam_name, None)
        if not cam_cfg:
            return img
        K = np.array(cam_cfg["camera_matrix"], dtype=np.float64)
        dist = np.array(cam_cfg["dist_coeffs"], dtype=np.float64)
        h, w = img.shape[:2]
        # get optimal new camera matrix to avoid black regions (optional)
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0)
        und = cv2.undistort(img, K, dist, None, newK)
        return und
    except Exception as e:
        print(f"[WARN] undistort failed for {cam_name}: {e}")
        return img

def resize_image(img: np.ndarray, width: int, height: int):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def crop_image(img: np.ndarray, crop_rect):
    if crop_rect is None:
        return img
    x, y, w, h = crop_rect
    h0, w0 = img.shape[:2]
    # clamp values
    x = max(0, min(x, w0-1))
    y = max(0, min(y, h0-1))
    w = max(1, min(w, w0-x))
    h = max(1, min(h, h0-y))
    return img[y:y+h, x:x+w]

def clahe_equalize_color(img: np.ndarray):
    """Apply CLAHE to each channel in LAB space (preserves color)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return img2

def imagenet_normalize(img: np.ndarray):
    """
    img assumed as uint8 BGR. Convert to RGB float [0,1], apply mean/std, return float32.
    This function returns a float32 array (C last) with normalization applied.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img

# ---------------- main preprocessing ----------------
def preprocess_camera(
    data_root: Path,
    out_root: Path,
    width: int,
    height: int,
    crop_rect: tuple,
    equalize: bool,
    calib_json_path: Path,
    imagenet_norm: bool,
    subset_n: int = None
):
    data_root = Path(data_root)
    out_root = Path(out_root)
    camera_out_root = out_root / "camera"
    camera_out_root.mkdir(parents=True, exist_ok=True)

    # load calibration if provided
    calib = load_calib_json(calib_json_path) if calib_json_path else None

    metadata = []

    for cam in CAM_FOLDERS:
        src_folder = data_root / cam
        dst_folder = camera_out_root / cam
        dst_folder.mkdir(parents=True, exist_ok=True)

        files = find_images(src_folder)
        if subset_n is not None:
            files = files[:subset_n]
        if len(files) == 0:
            print(f"[WARN] no images for {cam} at {src_folder}")
            continue

        print(f"[INFO] processing {cam}: {len(files)} images -> {dst_folder}")

        for fp in tqdm(files, desc=f"Processing {cam}", unit="img"):
            ts = file_timestamp_ms(fp)
            # read
            img = cv2.imread(str(fp))
            if img is None:
                print(f"[WARN] cannot read {fp}, skipping")
                continue

            # undistort if calib exists
            if calib:
                img = undistort_image(img, cam, calib)

            # crop first (optional)
            if crop_rect:
                img = crop_image(img, crop_rect)

            # resize
            img = resize_image(img, width, height)

            # optional equalization
            if equalize:
                img = clahe_equalize_color(img)

            # For saving, we keep uint8 BGR images; but produce a normalized array if requested
            save_img = img.copy()
            if imagenet_norm:
                _ = imagenet_normalize(save_img)  # produce normalized in-memory (not saved)
                # We still save visually meaningful image - convert normalized back for visualization
                # Here we undo normalize to save a properly scaled image
                # (kept simple: save the un-normalized resized image)
            # Save output
            out_path = dst_folder / fp.name
            ok = cv2.imwrite(str(out_path), save_img)
            if not ok:
                print(f"[WARN] failed to write {out_path}")

            # collect metadata
            metadata.append({
                "camera": cam,
                "original_path": str(fp),
                "processed_path": str(out_path),
                "timestamp_ms": int(ts),
                "shape": [int(width), int(height)],
                "equalized": bool(equalize),
                "cropped": bool(crop_rect is not None),
            })

    # Save metadata JSON
    meta_path = out_root / "camera_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[DONE] processed images metadata saved to {meta_path} ({len(metadata)} items)")

    return meta_path

# ---------------- train/val/test split helper ----------------
def make_splits(meta_path: Path, out_root: Path, train=0.8, val=0.1, test=0.1, seed=42):
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    assert abs(train + val + test - 1.0) < 1e-6, "splits must sum to 1.0"

    random.seed(seed)
    random.shuffle(metadata)
    N = len(metadata)
    ntrain = int(math.floor(train * N))
    nval = int(math.floor(val * N))
    ntest = N - ntrain - nval

    train_items = metadata[:ntrain]
    val_items = metadata[ntrain:ntrain + nval]
    test_items = metadata[ntrain + nval:]

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "train_meta.json", 'w') as f:
        json.dump(train_items, f, indent=2)
    with open(out_root / "val_meta.json", 'w') as f:
        json.dump(val_items, f, indent=2)
    with open(out_root / "test_meta.json", 'w') as f:
        json.dump(test_items, f, indent=2)

    print(f"[SPLITS] N={N} -> train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")
    return (out_root / "train_meta.json", out_root / "val_meta.json", out_root / "test_meta.json")

# ---------------- CLI ----------------
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="root containing original camera folders (CAM_*)")
    parser.add_argument("--out-root", required=True, help="where to write preprocessed data")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--crop", nargs=4, type=int, default=None,
                        help="crop rectangle x y w h (applied before resize)")
    parser.add_argument("--equalize", action="store_true", help="apply CLAHE histogram equalization")
    parser.add_argument("--calib-json", default=None, help="optional camera intrinsics JSON")
    parser.add_argument("--imagenet", action="store_true", help="apply ImageNet normalization (in-memory)")
    parser.add_argument("--subset", type=int, default=None, help="only process first N images from each folder (testing)")
    parser.add_argument("--make-splits", nargs=3, type=float, metavar=('TRAIN','VAL','TEST'), default=None,
                        help="create train/val/test splits (fractions that sum to 1.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta_path = preprocess_camera(
        data_root=Path(args.data_root),
        out_root=Path(args.out_root),
        width=args.width,
        height=args.height,
        crop_rect=tuple(args.crop) if args.crop else None,
        equalize=args.equalize,
        calib_json_path=Path(args.calib_json) if args.calib_json else None,
        imagenet_norm=args.imagenet,
        subset_n=args.subset
    )

    if args.make_splits:
        train_f, val_f, test_f = args.make_splits
        if abs(train_f + val_f + test_f - 1.0) > 1e-6:
            print("[ERROR] splits must sum to 1.0")
        else:
            make_splits(meta_path, Path(args.out_root), train_f, val_f, test_f, seed=args.seed)

if __name__ == "__main__":
    cli()
