#!/usr/bin/env python3
# scripts/train_depth.py
import argparse
import os
import json
from tqdm import tqdm
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# local imports (assumes repo root in PYTHONPATH or run from project root)
from src.models.radardepthnet import RadarDepthNet
from src.dataset.radarfusion_dataset import RadarFusionDataset

def compute_depth_metrics(pred, gt, valid_mask=None):
    """
    pred, gt: torch tensors [B,1,H,W]
    valid_mask: boolean mask [B,1,H,W] or None -> use gt>0 as mask
    returns aggregated (sum_rmse, sum_mae, count_pixels) and delta counts
    """
    with torch.no_grad():
        if valid_mask is None:
            mask = gt > 0
        else:
            mask = valid_mask
        # avoid empty mask
        mask_f = mask.float()
        valid = mask.bool()
        if valid.sum().item() == 0:
            return {'n':0, 'sse':0.0, 'mae':0.0, 'd1':0, 'd2':0, 'd3':0}
        diff = (pred - gt)
        sse = (diff[valid] ** 2).sum().item()
        mae = diff[valid].abs().sum().item()
        # delta metrics -- prevent division by zero by adding eps
        eps = 1e-6
        ratio = torch.max(pred[valid] / (gt[valid] + eps), gt[valid] / (pred[valid] + eps))
        d1 = (ratio < 1.25).sum().item()
        d2 = (ratio < 1.25**2).sum().item()
        d3 = (ratio < 1.25**3).sum().item()
        return {'n': int(valid.sum().item()), 'sse': sse, 'mae': mae, 'd1': d1, 'd2': d2, 'd3': d3}

def aggregate_metrics(list_of_dicts):
    agg = {'n':0,'sse':0.0,'mae':0.0,'d1':0,'d2':0,'d3':0}
    for d in list_of_dicts:
        for k in agg:
            agg[k] += d.get(k,0)
    if agg['n'] == 0:
        return {'rmse': float('nan'), 'mae': float('nan'), 'd1':0.0,'d2':0.0,'d3':0.0}
    rmse = math.sqrt(agg['sse'] / agg['n'])
    mae = agg['mae'] / agg['n']
    d1 = agg['d1'] / agg['n']
    d2 = agg['d2'] / agg['n']
    d3 = agg['d3'] / agg['n']
    return {'rmse': rmse, 'mae': mae, 'd1': d1, 'd2': d2, 'd3': d3}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Device:", device)

    # dataset + dataloader
    train_ds = RadarFusionDataset(args.train_meta, radar_points=args.radar_points)
    val_ds   = RadarFusionDataset(args.val_meta, radar_points=args.radar_points)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # model, optimizer
    model = RadarDepthNet(radar_dim=3, radar_points=args.radar_points).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_rmse = float('inf')

    # optionally resume
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print("Resumed from", args.resume, "start_epoch=", start_epoch)

    os.makedirs(args.out_dir, exist_ok=True)

    history = {'train_loss':[], 'val_rmse':[], 'val_mae':[]}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            radar = batch['radar'].to(device)
            gt = batch['dacc'].to(device)
            # optionally scale gt (if you used scaling)
            # forward
            pred = model(images, radar)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            loss = F.smooth_l1_loss(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (pbar.n + 1):.5f}")

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch+1} avg train loss: {avg_train_loss:.6f}")

        # validation
        model.eval()
        metric_list = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                radar = batch['radar'].to(device)
                gt = batch['dacc'].to(device)
                pred = model(images, radar)
                metric_list.append(compute_depth_metrics(pred, gt))
        val_metrics = aggregate_metrics(metric_list)
        print(f"Validation RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, d1: {val_metrics['d1']:.3f}")

        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])

        # save checkpoint every epoch and best
        ckpt = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_rmse': val_metrics['rmse']
        }
        ckpt_name = os.path.join(args.out_dir, f"radardepth_epoch{epoch+1}.pth")
        torch.save(ckpt, ckpt_name)
        print("Saved checkpoint:", ckpt_name)

        if val_metrics['rmse'] < best_rmse:
            best_rmse = val_metrics['rmse']
            torch.save(ckpt, os.path.join(args.out_dir, "radardepth_best.pth"))
            print("Saved best model.")

        # optionally adjust LR scheduler (not included by default)

        # write history json
        json.dump(history, open(os.path.join(args.out_dir, "train_history.json"), "w"), indent=2)

    print("Training finished. Best RMSE:", best_rmse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-meta", required=True, help="train metadata json (with dacc_path and radar)")
    parser.add_argument("--val-meta", required=True, help="val metadata json (with dacc_path and radar)")
    parser.add_argument("--out-dir", default="checkpoints", help="where to save checkpoints and history")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--radar-points", type=int, default=128)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--force-cpu", action="store_true", help="force CPU even if CUDA available")
    args = parser.parse_args()
    train(args)
