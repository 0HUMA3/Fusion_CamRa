#!/usr/bin/env python3
"""
Robust visualization for eval predictions.

Usage examples:

# Visualize specific pred file (npz or json)
python3 -m scripts.visualize_depth --pred-file eval_depth_results/preds/pred_b0_i0.npz --meta data/data/preprocessed_data/val_meta_fixedpaths_with_radar_with_dacc.json --out-dir depth_viz

# Visualize by index using meta + preds-dir
python3 -m scripts.visualize_depth --meta data/data/preprocessed_data/val_meta_fixedpaths_with_radar_with_dacc.json --preds-dir eval_depth_results/preds --index 0 --out-dir depth_viz
"""
import argparse
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import re

def choose_path_from_meta_field(maybe_path):
    """
    Accepts a string, list, or tuple. Returns a single path string or None.
    If maybe_path is a list/tuple, return the first element that looks like a path.
    """
    if maybe_path is None:
        return None
    if isinstance(maybe_path, (list, tuple)):
        # pick the first element that's a string-like and non-empty
        for p in maybe_path:
            if isinstance(p, str) and p:
                return p
        return None
    if isinstance(maybe_path, str):
        return maybe_path
    # fallback: convert to str if possible
    try:
        return str(maybe_path)
    except Exception:
        return None

def load_image(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr  # H,W,3 float

def load_npz_array(path, prefer_key=None):
    if path.endswith('.npz'):
        data = np.load(path, allow_pickle=True)
        if prefer_key and prefer_key in data.files:
            return data[prefer_key]
        for k in ('pred','depth','arr_0','arr'):
            if k in data.files:
                return data[k]
        if data.files:
            return data[data.files[0]]
        raise RuntimeError(f"No arrays in {path}")
    else:
        return np.load(path, allow_pickle=True)

def ensure_hw_match(img, arr):
    if arr is None:
        return None
    a = arr
    if isinstance(a, np.ndarray) and a.ndim == 3 and a.shape[0] == 1:
        a = a.squeeze(0)
    if isinstance(a, np.ndarray) and a.ndim == 3 and a.shape[2] == 3:
        a = a.mean(axis=2)
    Himg, Wimg = img.shape[0], img.shape[1]
    if a.shape[0] != Himg or a.shape[1] != Wimg:
        # resize via PIL using normalized uint8
        a_min = float(np.nanmin(a)) if np.isfinite(a).any() else 0.0
        a_max = float(np.nanmax(a)) if np.isfinite(a).any() else 1.0
        if a_max > a_min:
            norm = (a - a_min) / (a_max - a_min)
            to_resize = (np.clip(norm,0,1) * 255.0).astype(np.uint8)
            pil = Image.fromarray(to_resize)
            pil = pil.resize((Wimg, Himg), resample=Image.BILINEAR)
            resized = np.asarray(pil).astype(np.float32) / 255.0
            out = resized * (a_max - a_min) + a_min
        else:
            pil = Image.fromarray((a.astype(np.float32)).astype(np.uint8))
            pil = pil.resize((Wimg, Himg), resample=Image.BILINEAR)
            out = np.asarray(pil).astype(np.float32)
        return out
    return a

def make_viz(image, gt_depth, pred_depth, outpath, cmap='magma'):
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    ax = axes[0]; ax.imshow(np.clip(image,0,1)); ax.axis('off'); ax.set_title("RGB")
    ax = axes[1]; im1 = ax.imshow(gt_depth, cmap=cmap); ax.set_title("GT depth"); ax.axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    ax = axes[2]; im2 = ax.imshow(pred_depth, cmap=cmap); ax.set_title("Pred depth"); ax.axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)

def overlay_on_image(image, depth, max_depth=None, alpha=0.6, cmap='magma'):
    import matplotlib.cm as cm
    if max_depth is None:
        if np.isfinite(depth).any():
            max_depth = float(np.nanmax(depth[np.isfinite(depth)]))
        else:
            max_depth = 1.0
    if max_depth <= 0:
        max_depth = 1.0
    norm = np.clip(depth / (max_depth + 1e-6), 0.0, 1.0)
    col = plt.get_cmap(cmap)(norm)[:,:,:3]
    overlay = (1.0-alpha)*image + alpha*col
    return np.clip(overlay,0,1)

def parse_ts_from_basename(fn):
    groups = re.findall(r'(\d{8,})', fn)
    if not groups:
        return None
    try:
        return int(groups[-1])
    except:
        return None

def find_meta_for_pred(pred_json, meta_list):
    # 1) if pred_json contains meta, use it
    if pred_json and isinstance(pred_json, dict) and pred_json.get('meta'):
        return pred_json['meta']
    # else try to match by processed_path basename substring
    predname = pred_json.get('pred_path') if pred_json else None
    if not predname:
        return None
    bn = os.path.basename(predname)
    key = os.path.splitext(bn)[0]
    for it in meta_list:
        proc = it.get('processed_path') or it.get('processed_image') or it.get('processed')
        if proc and key in os.path.basename(proc):
            return it
    # try timestamp matching
    p_ts = parse_ts_from_basename(key)
    if p_ts:
        # exact match by timestamp
        for it in meta_list:
            ts = it.get('timestamp_ms')
            if ts and abs(int(ts) - p_ts) < 1000:  # within 1s
                return it
        # fallback: find nearest
        best=None; best_diff=10**18
        for it in meta_list:
            ts = it.get('timestamp_ms')
            if ts:
                d = abs(int(ts) - p_ts)
                if d < best_diff:
                    best_diff=d; best=it
        return best
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", help="meta json (optional, used to find GT/image)", default=None)
    parser.add_argument("--preds-dir", help="folder with preds (.npz/.json)", default=None)
    parser.add_argument("--pred-file", help="single pred file (.npz or .json) to visualize", default=None)
    parser.add_argument("--index", type=int, default=0, help="index into meta (if using meta+preds-dir)")
    parser.add_argument("--out-dir", default="viz_out")
    parser.add_argument("--cmap", default="magma")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_list = None
    if args.meta:
        meta_list = json.load(open(args.meta))

    # If using preds-dir + index:
    if not args.pred_file and args.preds_dir and meta_list:
        idx = args.index
        if idx < 0 or idx >= len(meta_list):
            raise SystemExit(f"index out of range {idx} not in [0,{len(meta_list)-1}]")
        it = meta_list[idx]
        img_path = choose_path_from_meta_field(it.get('processed_path') or it.get('processed_image') or it.get('processed'))

        if not img_path or not os.path.exists(img_path):
            raise SystemExit(f"Image not found for index {idx}: {img_path}")
        image = load_image(img_path)
        # gt
        dacc_path = it.get('dacc_path') or it.get('dacc') or it.get('dacc_file')
        if dacc_path and os.path.exists(dacc_path):
            gt_arr = load_npz_array(dacc_path, prefer_key='depth')
        else:
            print("Warning: GT dacc missing, GT will be zeros")
            gt_arr = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        # find corresponding pred file by searching preds_dir for substring match of processed basename
        proc_bn = os.path.splitext(os.path.basename(img_path))[0]
        pred_file = None
        for fn in os.listdir(args.preds_dir):
            if proc_bn in fn and fn.endswith(('.npz','.npy','.json')):
                pred_file = os.path.join(args.preds_dir, fn); break
        if pred_file is None:
            # fallback: pick first pred
            files = sorted([os.path.join(args.preds_dir,f) for f in os.listdir(args.preds_dir) if f.endswith(('.npz','.npy','.json'))])
            if not files:
                raise SystemExit("No pred files found in preds_dir")
            pred_file = files[0]
            print("Warning: no exact match found; using first pred file:", pred_file)
        # load pred
        if pred_file.endswith('.json'):
            pd = json.load(open(pred_file)); pp = pd.get('pred_path') or pred_file.replace('.json','.npz')
            pred_arr = load_npz_array(pp, prefer_key='pred')
        else:
            # try to locate sibling .json with meta
            sibling_json = pred_file[:-4]+'.json' if pred_file.endswith('.npz') else None
            pred_json = None
            if sibling_json and os.path.exists(sibling_json):
                try:
                    pred_json = json.load(open(sibling_json))
                except:
                    pred_json = None
            if pred_json and pred_json.get('pred_path'):
                pred_arr = load_npz_array(pred_json['pred_path'], prefer_key='pred')
            else:
                pred_arr = load_npz_array(pred_file, prefer_key='pred')
        pred_depth = pred_arr.squeeze()
        gt_depth = gt_arr.squeeze()
        pred_depth = ensure_hw_match(image, pred_depth)
        gt_depth = ensure_hw_match(image, gt_depth)
        out_png = os.path.join(args.out_dir, f"viz_idx{idx}_{proc_bn}.png")
        make_viz(image, gt_depth, pred_depth, out_png, cmap=args.cmap)
        overlay = overlay_on_image(image, pred_depth, cmap=args.cmap)
        overlay_path = os.path.join(args.out_dir, f"overlay_idx{idx}_{proc_bn}.png")
        Image.fromarray((overlay*255).astype(np.uint8)).save(overlay_path)
        print("Saved:", out_png, overlay_path)
        return

    # If pred-file provided directly:
    if args.pred_file:
        pred_path = args.pred_file
        raw_pred_json = None
        # if .json, try to read
        if pred_path.endswith('.json'):
            try:
                raw_pred_json = json.load(open(pred_path))
                if raw_pred_json.get('pred_path'):
                    pred_path = raw_pred_json['pred_path']
            except:
                raw_pred_json = None
        # try to find sibling JSON if pred_path is npz
        if pred_path.endswith('.npz') and os.path.exists(pred_path[:-4]+'.json'):
            try:
                raw_pred_json = json.load(open(pred_path[:-4]+'.json'))
            except:
                raw_pred_json = raw_pred_json
        # attempt to find meta via pred-json and supplied meta list
        matched_meta = None
        if raw_pred_json and meta_list:
            matched_meta = find_meta_for_pred(raw_pred_json, meta_list)
        # if no meta yet, try to match by pred filename basename in meta
        if not matched_meta and meta_list:
            bn = os.path.splitext(os.path.basename(pred_path))[0]
            for it in meta_list:
                proc = it.get('processed_path') or it.get('processed_image') or it.get('processed')
                if proc and bn in os.path.basename(proc):
                    matched_meta = it; break
            # timestamp fallback
            if not matched_meta:
                matched_meta = find_meta_for_pred({'pred_path': os.path.basename(pred_path)}, meta_list)
        # load pred array
        pred_arr = load_npz_array(pred_path, prefer_key='pred')
        pred_depth = pred_arr.squeeze()
        if matched_meta:
            img_path = choose_path_from_meta_field(matched_meta.get('processed_path') or matched_meta.get('processed_image') or matched_meta.get('processed'))

            if img_path and os.path.exists(img_path):
                image = load_image(img_path)
            else:
                print("Warning: matched meta but image missing, creating black image")
                # create placeholder based on pred size
                if pred_depth.ndim==2:
                    image = np.zeros((pred_depth.shape[0], pred_depth.shape[1], 3), dtype=np.float32)
                else:
                    image = np.zeros((pred_depth.shape[1], pred_depth.shape[2], 3), dtype=np.float32)
            dacc_path = matched_meta.get('dacc_path') or matched_meta.get('dacc') or matched_meta.get('dacc_file')
            if dacc_path and os.path.exists(dacc_path):
                gt_arr = load_npz_array(dacc_path, prefer_key='depth')
            else:
                print("Warning: dacc not present; GT will be zeros.")
                if pred_depth.ndim==2:
                    gt_arr = np.zeros_like(pred_depth)
                else:
                    gt_arr = np.zeros((pred_depth.shape[1], pred_depth.shape[2]))
            gt_depth = gt_arr.squeeze()
            pred_depth = ensure_hw_match(image, pred_depth)
            gt_depth = ensure_hw_match(image, gt_depth)
            out_png = os.path.join(args.out_dir, f"viz_pred_{os.path.basename(pred_path)}.png")
            make_viz(image, gt_depth, pred_depth, out_png, cmap=args.cmap)
            overlay = overlay_on_image(image, pred_depth, cmap=args.cmap)
            overlay_path = os.path.join(args.out_dir, f"overlay_{os.path.basename(pred_path)}.png")
            Image.fromarray((overlay*255).astype(np.uint8)).save(overlay_path)
            print("Saved:", out_png, overlay_path)
            return
        # last resort: visualize pred only (no GT)
        # synthesize black image sized to pred
        if pred_depth.ndim==2:
            H,W = pred_depth.shape
        elif pred_depth.ndim==3 and pred_depth.shape[0]==1:
            H,W = pred_depth.shape[1], pred_depth.shape[2]
            pred_depth = pred_depth.squeeze(0)
        else:
            # try infer
            H,W = pred_depth.shape[-2], pred_depth.shape[-1]
            pred_depth = pred_depth.reshape(H,W)
        image = np.zeros((H,W,3), dtype=np.float32)
        gt_depth = np.zeros_like(pred_depth)
        out_png = os.path.join(args.out_dir, f"viz_pred_{os.path.basename(pred_path)}.png")
        make_viz(image, gt_depth, pred_depth, out_png, cmap=args.cmap)
        overlay = overlay_on_image(image, pred_depth, cmap=args.cmap)
        overlay_path = os.path.join(args.out_dir, f"overlay_{os.path.basename(pred_path)}.png")
        Image.fromarray((overlay*255).astype(np.uint8)).save(overlay_path)
        print("Saved (no meta):", out_png, overlay_path)
        return

    # fallback error
    raise SystemExit("Provide either --pred-file or both --meta + --preds-dir + --index")

if __name__ == "__main__":
    main()
