#!/usr/bin/env python3
import os
import glob
import numpy as np
import argparse

# Optional open3d import will be done inside functions to allow the script to warn gracefully
VOXEL_SIZE = 0.5            # example for down-sampling (in meters)
MAX_RANGE = 50.0            # filter out radar points farther than this (in meters)

def process_one_file(in_path, out_path, create_dirs=False):
    """Read one .pcd file, filter points, save as .npy."""
    try:
        import open3d as o3d
    except Exception:
        raise RuntimeError("Open3D is required for radar preprocessing. Install open3d.")
    pcd = o3d.io.read_point_cloud(in_path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        print(f"WARNING: no points read from {in_path}")
        return
    # filter by range in XY plane
    dists = np.linalg.norm(pts[:, :2], axis=1)
    mask = dists < MAX_RANGE
    pts = pts[mask]
    # optional down‐sample via voxel grid
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts)
    down = pcd2.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pts_down = np.asarray(down.points)
    # ensure output directory exists only if create_dirs True
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        if create_dirs:
            os.makedirs(out_dir, exist_ok=True)
        else:
            print(f"Skipping save for {out_path} because directory does not exist. Use --create-dirs to create it.")
            return
    np.save(out_path, pts_down)
    print(f"Saved: {out_path}  –  shape: {pts_down.shape}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/data', help='root folder containing original_data and preprocessed_data')
    p.add_argument('--voxel-size', type=float, default=VOXEL_SIZE, help='voxel size for downsampling')
    p.add_argument('--max-range', type=float, default=MAX_RANGE, help='max range to keep radar points')
    p.add_argument('--create-dirs', action='store_true', help='Create output directories if missing')
    args = p.parse_args()

    global VOXEL_SIZE, MAX_RANGE
    VOXEL_SIZE = args.voxel_size
    MAX_RANGE = args.max_range

    ROOT = args.root
    ORIG_RADAR_DIR = os.path.join(ROOT, "original_data")
    PREP_RADAR_DIR = os.path.join(ROOT, "preprocessed_data", "radar")

    radar_dirs = ["RADAR_BACK_LEFT", "RADAR_BACK_RIGHT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
    for rd in radar_dirs:
        in_folder = os.path.join(ORIG_RADAR_DIR, rd)
        pattern = os.path.join(in_folder, "*.pcd")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No files found in {in_folder}")
            continue
        print(f"Processing folder {rd}: found {len(files)} files")
        for fpath in files:
            fname = os.path.basename(fpath)
            name_no_ext = os.path.splitext(fname)[0]
            out_fname = name_no_ext + ".npy"
            out_path = os.path.join(PREP_RADAR_DIR, rd, out_fname)
            try:
                process_one_file(fpath, out_path, create_dirs=args.create_dirs)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")

if __name__ == "__main__":
    main()
