#!/usr/bin/env python3
# scripts/eval_depth.py

import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.radardepthnet import RadarDepthNet
from src.dataset.radarfusion_dataset import RadarFusionDataset
from torch.utils.data import DataLoader


# ============================
# Metric Functions
# ============================
def compute_depth_metrics(pred, gt, valid_mask=None):
    with torch.no_grad():
        if valid_mask is None:
            mask = gt > 0
        else:
            mask = valid_mask
        if mask.sum().item() == 0:
            return {'n': 0, 'sse': 0.0, 'mae': 0.0, 'd1': 0, 'd2': 0, 'd3': 0}

        diff = (pred - gt)
        sse = (diff[mask] ** 2).sum().item()
        mae = diff[mask].abs().sum().item()
        eps = 1e-6
        ratio = torch.max(pred[mask] / (gt[mask] + eps), gt[mask] / (pred[mask] + eps))
        d1 = (ratio < 1.25).sum().item()
        d2 = (ratio < 1.25 ** 2).sum().item()
        d3 = (ratio < 1.25 ** 3).sum().item()
        return {
            'n': int(mask.sum().item()),
            'sse': sse,
            'mae': mae,
            'd1': int(d1),
            'd2': int(d2),
            'd3': int(d3),
        }


def aggregate_metrics(list_of_dicts):
    agg = {'n': 0, 'sse': 0.0, 'mae': 0.0, 'd1': 0, 'd2': 0, 'd3': 0}
    for d in list_of_dicts:
        for k in agg:
            agg[k] += d.get(k, 0)
    if agg['n'] == 0:
        return {'rmse': float('nan'), 'mae': float('nan'), 'd1': 0.0, 'd2': 0.0, 'd3': 0.0}
    rmse = (agg['sse'] / agg['n']) ** 0.5
    mae = agg['mae'] / agg['n']
    return {
        'rmse': rmse,
        'mae': mae,
        'd1': agg['d1'] / agg['n'],
        'd2': agg['d2'] / agg['n'],
        'd3': agg['d3'] / agg['n'],
    }


# ============================
# JSON Sanitization Helper
# ============================
def sanitize_for_json(obj):
    """
    Convert torch tensors, numpy arrays, and other objects
    into JSON-safe types recursively.
    """
    # torch tensor
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        return arr.tolist() if arr.ndim > 0 else arr.item()

    # numpy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
        return float(obj)

    # dict / list
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]

    # simple types
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # fallback
    return str(obj)


# ============================
# Visualization Helper
# ============================
def make_viz(image, gt_depth, pred_depth, outpath):
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.cpu().squeeze(0).numpy()
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.cpu().squeeze(0).numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(image, 0, 1))
    plt.axis('off')
    plt.title('RGB')

    plt.subplot(1, 3, 2)
    plt.imshow(gt_depth, cmap='magma')
    plt.title('GT')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(pred_depth, cmap='magma')
    plt.title('Pred')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default="eval_results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--radar-points", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-max", type=int, default=8, help="max visualizations to save")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preds_dir = os.path.join(args.out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    ds = RadarFusionDataset(args.meta, radar_points=args.radar_points)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = RadarDepthNet(radar_dim=3, radar_points=args.radar_points).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    metric_list = []
    viz_saved = 0

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(dl, desc="Eval")):
            images = batch['image'].to(device)
            radar = batch['radar'].to(device)
            gt = batch['dacc'].to(device)
            preds = model(images, radar)   # [B,1,H,W]

            for i in range(preds.shape[0]):
                p = preds[i:i+1]
                g = gt[i:i+1]
                metric_list.append(compute_depth_metrics(p, g))

                # Save prediction
                out_json = {'pred_path': None, 'meta': None}
                fname = f"pred_b{bidx}_i{i}.npz"
                np_out_path = os.path.join(preds_dir, fname)
                np.savez_compressed(np_out_path, pred=p.cpu().numpy())
                out_json['pred_path'] = np_out_path

                meta = batch.get('meta', None)
                if meta:
                    if isinstance(meta, (list, tuple)):
                        raw_meta = meta[i]
                    else:
                        raw_meta = meta
                    out_json['meta'] = sanitize_for_json(raw_meta)

                json.dump(
                    out_json,
                    open(os.path.join(preds_dir, f"pred_b{bidx}_i{i}.json"), "w"),
                    indent=2
                )

                # Optional visualization
                if args.visualize and viz_saved < args.viz_max:
                    try:
                        img_np = images[i].cpu()
                        gt_np = gt[i].cpu()
                        pred_np = preds[i].cpu()
                        viz_path = os.path.join(args.out_dir, f"viz_b{bidx}_i{i}.png")
                        make_viz(img_np, gt_np, pred_np, viz_path)
                        viz_saved += 1
                    except Exception as e:
                        print("Visualization failed:", e)

    # Final aggregation
    agg = aggregate_metrics(metric_list)
    print("Eval metrics:", agg)
    json.dump(agg, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)
    print("Saved preds to", preds_dir)


if __name__ == "__main__":
    main()
