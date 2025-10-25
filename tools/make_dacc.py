#!/usr/bin/env python3
"""
tools/make_dacc.py

Generate accumulated static depth (dacc) for camera frames.

Modes:
 - nuscenes: use NuScenes API to fetch calibrated transforms and lidar sweeps,
             reproject to camera frame and z-buffer.
 - files:    find nearby LiDAR files in a folder by timestamp and accumulate them.

This version requires explicit --create-dirs to create output folders.
"""

import os, sys, argparse, glob, time
import numpy as np
from tqdm import tqdm
from math import inf

# Optional imports for nuscenes mode
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    NUSCENES_OK = True
except Exception:
    NUSCENES_OK = False

# Optional open3d for .pcd reading in files mode
try:
    import open3d as o3d
    O3D_OK = True
except Exception:
    O3D_OK = False

# ---------------------------
# small helpers
# ---------------------------
def inv4(T):
    R = T[:3,:3]
    t = T[:3,3]
    Rinv = R.T
    tinv = -Rinv @ t
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3,:3] = Rinv
    Tinv[:3,3] = tinv
    return Tinv

def pts_hom(pts):
    n = pts.shape[0]
    h = np.ones((n,4), dtype=pts.dtype)
    h[:,:3] = pts
    return h

def transform_points(T, pts):
    ph = pts_hom(pts)
    out = (T @ ph.T).T
    return out[:,:3]

def project_camera(K, pts_cam):
    # pts_cam: Nx3 (X,Y,Z) camera coords
    Z = pts_cam[:,2]
    mask = Z > 1e-4
    if np.sum(mask) == 0:
        return np.array([]), np.array([]), np.array([])
    X = pts_cam[mask,0]; Y = pts_cam[mask,1]; Zp = pts_cam[mask,2]
    u = (K[0,0]*X + K[0,2]*Zp) / Zp
    v = (K[1,1]*Y + K[1,2]*Zp) / Zp
    return u, v, Zp

def points_to_depth_map(u, v, z, H, W):
    depth = np.zeros((H,W), dtype=np.float32)
    xi = np.round(u).astype(np.int32)
    yi = np.round(v).astype(np.int32)
    mask = (xi>=0)&(xi<W)&(yi>=0)&(yi<H)
    xi = xi[mask]; yi = yi[mask]; z = z[mask]
    for x,y,zz in zip(xi, yi, z):
        cur = depth[y,x]
        if cur == 0 or zz < cur:
            depth[y,x] = zz
    return depth

# ---------------------------
# nuscenes mode implementations
# ---------------------------
def make_dacc_nuscenes(nusc_root, cam_name, out_dir, window, max_frames, mask_map=None, create_dirs=False):
    if not NUSCENES_OK:
        raise RuntimeError("NuScenes python package not available. Install nuscenes-devkit.")
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root, verbose=False)

    if create_dirs:
        os.makedirs(out_dir, exist_ok=True)
    else:
        if not os.path.exists(out_dir):
            print(f"Warning: out_dir '{out_dir}' does not exist. Run with --create-dirs to create it. Files will be skipped.")

    # gather camera sample_data tokens for chosen cam_name
    cam_samples = []
    for sd in nusc.sample_data:
        if sd['sensor_modality'] == 'camera' and sd['channel'] == cam_name:
            cam_samples.append(sd)
    print("Found camera sample_data entries:", len(cam_samples))

    # sort by timestamp
    cam_samples = sorted(cam_samples, key=lambda s: s['timestamp'])
    total = 0
    for i, cam_sd in enumerate(tqdm(cam_samples)):
        if max_frames and total >= max_frames:
            break
        ts = cam_sd['timestamp']
        sample = nusc.get('sample', cam_sd['sample_token'])
        lidar_sds = [s for s in nusc.sample_data if s['sensor_modality']=='lidar' and s['channel']=='LIDAR_TOP']
        diffs = [(abs(s['timestamp'] - ts), s) for s in lidar_sds]
        diffs = sorted(diffs, key=lambda x: x[0])
        chosen = [d[1] for d in diffs[:window]]

        cam_cal = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_cam_intrinsic = np.array(cam_cal['camera_intrinsic']).reshape(3,3)
        T_cam_s2e = np.eye(4); T_cam_s2e[:3,:3] = Quaternion(cam_cal['rotation']).rotation_matrix; T_cam_s2e[:3,3] = cam_cal['translation']
        cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
        T_cam_e2w = np.eye(4); T_cam_e2w[:3,:3] = Quaternion(cam_pose['rotation']).rotation_matrix; T_cam_e2w[:3,3] = cam_pose['translation']
        T_cam_to_world = T_cam_e2w @ T_cam_s2e
        T_world_to_cam = inv4(T_cam_to_world)

        # infer image size by loading the image size from file
        img_rel_path = cam_sd['filename']
        img_abs = os.path.join(nusc_root, img_rel_path)
        from PIL import Image, ImageOps
        img = Image.open(img_abs); img = ImageOps.exif_transpose(img)
        W, H = img.size[0], img.size[1]
        depth_accum = np.zeros((H,W), dtype=np.float32)

        for sd_lidar in chosen:
            lidar_path = os.path.join(nusc_root, sd_lidar['filename'])
            pc = LidarPointCloud.from_file(lidar_path)
            pts = pc.points[:3].T
            lidar_cal = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
            T_src_s2e = np.eye(4); T_src_s2e[:3,:3] = Quaternion(lidar_cal['rotation']).rotation_matrix; T_src_s2e[:3,3] = lidar_cal['translation']
            lidar_pose = nusc.get('ego_pose', sd_lidar['ego_pose_token'])
            T_src_e2w = np.eye(4); T_src_e2w[:3,:3] = Quaternion(lidar_pose['rotation']).rotation_matrix; T_src_e2w[:3,3] = lidar_pose['translation']
            T_src_to_world = T_src_e2w @ T_src_s2e
            T_src_to_cam = T_world_to_cam @ T_src_to_world
            pts_cam = transform_points(T_src_to_cam, pts)
            u, v, z = project_camera(cam_cam_intrinsic, pts_cam)
            if z.size == 0:
                continue
            depth_map = points_to_depth_map(u, v, z, H, W)
            mask_new = depth_map > 0
            mask_old = depth_accum > 0
            depth_accum[mask_new & ~mask_old] = depth_map[mask_new & ~mask_old]
            both = mask_new & mask_old
            depth_accum[both] = np.minimum(depth_accum[both], depth_map[both])

        base = os.path.basename(img_abs)
        bn = os.path.splitext(base)[0]
        if mask_map and bn in mask_map and mask_map[bn] is not None:
            mpath = mask_map[bn]
            if os.path.exists(mpath):
                mov = np.load(mpath)
                depth_accum[mov] = 0.0

        outfile = os.path.join(out_dir, bn + ".npz")
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            if create_dirs:
                os.makedirs(outdir, exist_ok=True)
            else:
                print(f"Skipping write of {outfile} because directory does not exist. Use --create-dirs to create it.")
                continue

        np.savez_compressed(outfile, depth=depth_accum, valid=(depth_accum>0))
        total += 1
    print("Finished. Wrote", total, "dacc files to", out_dir)

# ---------------------------
# files mode: naive accumulation by timestamp matching
# ---------------------------
def parse_ts_from_basename(fn):
    import re, os
    bn = os.path.basename(fn)
    groups = re.findall(r'(\d{8,})', bn)
    if not groups:
        return None
    try:
        return int(groups[-1])
    except:
        return None

def load_pcd_or_npy(path):
    ext = os.path.splitext(path)[1].lower()
    if path.lower().endswith('.pcd.bin'):
        try:
            data = np.fromfile(path, dtype=np.float32)
            if data.size == 0:
                return np.zeros((0,3), dtype=np.float32)
            pts = data.reshape((-1, 4))[:, :3]
            return pts.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to read .pcd.bin file {path}: {e}")

    if ext == '.npy':
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float32)
        else:
            raise RuntimeError("Loaded .npy has unexpected shape: " + str(arr.shape))

    if ext in ('.pcd', '.ply'):
        if not O3D_OK:
            raise RuntimeError("Open3D not installed. Install open3d or convert .pcd to .npy/.pcd.bin")
        import open3d as o3d_local
        pcd = o3d_local.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            return np.zeros((0,3), dtype=np.float32)
        return pts.astype(np.float32)

    raise RuntimeError("Unsupported point cloud format: " + ext + " (path: " + path + ")")

def make_dacc_files_mode(lidar_folder, cam_folder, out_dir, window, max_frames, mask_map=None, create_dirs=False):
    if create_dirs:
        os.makedirs(out_dir, exist_ok=True)
    else:
        if not os.path.exists(out_dir):
            print(f"Warning: out_dir '{out_dir}' does not exist. Run with --create-dirs to create it. Files will be skipped.")

    cam_images = sorted(glob.glob(os.path.join(cam_folder, "*.jpg")) + glob.glob(os.path.join(cam_folder, "*.png")))
    lidar_files = []
    lidar_files += sorted(glob.glob(os.path.join(lidar_folder, "*.pcd")))
    lidar_files += sorted(glob.glob(os.path.join(lidar_folder, "*.pcd.bin")))
    lidar_files += sorted(glob.glob(os.path.join(lidar_folder, "*.npy")))
    lidar_files += sorted(glob.glob(os.path.join(lidar_folder, "*.ply")))
    lidar_files = sorted(list(dict.fromkeys(lidar_files)))

    lidar_ts = []
    for f in lidar_files:
        t = parse_ts_from_basename(f)
        if t is not None:
            lidar_ts.append((t,f))
    if len(lidar_ts)==0:
        print("No timestamps parsed from lidar files. Aborting.")
        return

    total = 0
    images_iter = cam_images[:max_frames] if max_frames else cam_images
    for img_path in tqdm(images_iter):
        bn = os.path.splitext(os.path.basename(img_path))[0]
        ts = parse_ts_from_basename(img_path)
        if ts is None:
            continue
        diffs = [(abs(ts - lt), lf) for lt, lf in lidar_ts]
        diffs = sorted(diffs, key=lambda x: x[0])[:window]
        all_pts = []
        for d,lf in diffs:
            pts = load_pcd_or_npy(lf)
            all_pts.append(pts)
        if len(all_pts)==0:
            continue
        pts_all = np.vstack(all_pts)
        from PIL import Image, ImageOps
        img = Image.open(img_path); img = ImageOps.exif_transpose(img)
        W, H = img.size[0], img.size[1]
        fx = fy = 700.0
        K = np.array([[fx, 0, W/2.0],[0, fy, H/2.0],[0,0,1]])
        u, v, z = project_camera(K, pts_all)
        depth_map = points_to_depth_map(u, v, z, H, W)
        if mask_map and bn in mask_map and mask_map[bn] is not None:
            mpath = mask_map[bn]
            if os.path.exists(mpath):
                mov = np.load(mpath)
                depth_map[mov] = 0.0
        out = os.path.join(out_dir, bn + ".npz")
        outdir = os.path.dirname(out)
        if not os.path.exists(outdir):
            if create_dirs:
                os.makedirs(outdir, exist_ok=True)
            else:
                print(f"Skipping write of {out} because directory does not exist. Use --create-dirs to create it.")
                continue
        np.savez_compressed(out, depth=depth_map, valid=(depth_map>0))
        total += 1
    print("Finished. Wrote", total, "dacc files to", out_dir)

# ---------------------------
# main CLI
# ---------------------------
def build_mask_map_from_meta(meta_file):
    if not meta_file:
        return None
    import json
    with open(meta_file) as f:
        data = json.load(f)
    mm = {}
    for it in data:
        proc = it.get('processed_path') or it.get('original_path') or ''
        bn = os.path.splitext(os.path.basename(proc))[0]
        mpath = it.get('mask_path')
        mm[bn] = mpath
    return mm

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['nuscenes','files'], default='files')
    p.add_argument('--nuscenes-root', default='data/nuscenes')
    p.add_argument('--cam-name', default='CAM_FRONT_LEFT', help='channel name in nuscenes (CAM_FRONT_LEFT etc.)')
    p.add_argument('--lidar-folder', default='data/data/original_data/LIDAR_TOP')
    p.add_argument('--cam-folder', default='data/data/original_data/CAM_FRONT_LEFT')
    p.add_argument('--out-dir', default='data/data/preprocessed_data/dacc')
    p.add_argument('--window', type=int, default=5, help='number of nearest lidar sweeps to accumulate')
    p.add_argument('--max-frames', type=int, default=0, help='limit frames (0 = all)')
    p.add_argument('--meta-file', default='data/data/preprocessed_data/train_meta_fixedpaths_with_radar_with_masks.json')
    p.add_argument('--create-dirs', action='store_true', help='Create output directories if missing')
    args = p.parse_args()

    mask_map = build_mask_map_from_meta(args.meta_file)

    if args.mode == 'nuscenes':
        print("Running in nuscenes mode. Requires nuscenes-devkit and dataroot:", args.nuscenes_root)
        if not NUSCENES_OK:
            print("NuScenes package not found. pip install nuscenes-devkit")
            sys.exit(1)
        make_dacc_nuscenes(args.nuscenes_root, args.cam_name, args.out_dir, args.window, args.max_frames, mask_map=mask_map, create_dirs=args.create_dirs)
    else:
        print("Running in files mode. Lidar folder:", args.lidar_folder)
        make_dacc_files_mode(args.lidar_folder, args.cam_folder, args.out_dir, args.window, args.max_frames, mask_map=mask_map, create_dirs=args.create_dirs)

if __name__ == '__main__':
    main()
