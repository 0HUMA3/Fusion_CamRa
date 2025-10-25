#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
from PIL import Image, ImageOps
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

def read_image(fn):
    img = Image.open(fn)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return np.asarray(img)

def get_moving_ids(segments_info):
    # same heuristic used in the repo: COCO IDs 0..8 are moving object categories
    mv = []
    for s in segments_info:
        if s.get("category_id", -1) >= 0 and s.get("category_id", -1) <= 8:
            mv.append(s["id"])
    return mv

def main(args):
    # detectron2 setup
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    # use pretrained weights from detectron2 model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
    predictor = DefaultPredictor(cfg)

    input_root = args.input_root  # e.g. data/data/original_data
    output_dir = args.output_dir  # e.g. data/data/derived_masks

    # NOTE: only create directories if user explicitly asked for it
    if args.create_dirs:
        os.makedirs(output_dir, exist_ok=True)
    else:
        if not os.path.exists(output_dir):
            print(f"Warning: output_dir '{output_dir}' does not exist. Run with --create-dirs to create it.")
            # we will still continue but skip saving masks if dir is missing

    # iterate camera folders
    if not os.path.exists(input_root) or not os.path.isdir(input_root):
        print(f"Error: input_root '{input_root}' does not exist or is not a directory. Exiting.")
        return

    cam_folders = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])
    print("Found camera subfolders:", cam_folders)

    count = 0
    for sub in cam_folders:
        subpath = os.path.join(input_root, sub)
        image_paths = sorted(glob.glob(os.path.join(subpath, "*.jpg")) + glob.glob(os.path.join(subpath, "*.png")))
        print(f"{sub}: {len(image_paths)} images")
        for img_path in image_paths:
            if args.max_images and count >= args.max_images:
                print("Reached max_images, exiting.")
                return
            try:
                img = read_image(img_path)
                outputs = predictor(img)
                # outputs["panoptic_seg"] is (panoptic_seg, segments_info)
                pan_seg, segments_info = outputs["panoptic_seg"]
                # pan_seg may be a torch.Tensor (H,W) where values are segment ids
                if isinstance(pan_seg, torch.Tensor):
                    pan_seg_np = pan_seg.cpu().numpy()
                else:
                    pan_seg_np = np.asarray(pan_seg)

                moving_ids = get_moving_ids(segments_info)
                if len(moving_ids) == 0:
                    mask = np.zeros_like(pan_seg_np, dtype=bool)
                else:
                    mask = np.isin(pan_seg_np, moving_ids)

                # save mask
                base = os.path.basename(img_path)
                outname = os.path.splitext(base)[0] + ".npy"
                outpath = os.path.join(output_dir, outname)

                outdir = os.path.dirname(outpath)
                if not os.path.exists(outdir):
                    if args.create_dirs:
                        os.makedirs(outdir, exist_ok=True)
                    else:
                        print(f"Skipping save for {outpath} because directory does not exist. Use --create-dirs to allow creation.")
                        continue

                np.save(outpath, mask.astype(np.bool_))
                count += 1
                if count % 200 == 0:
                    print(f"Processed {count} images")
            except Exception as e:
                print("Error processing", img_path, e)
    print("Done. Total processed:", count)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", default="data/data/original_data", help="root with camera subfolders")
    p.add_argument("--output-dir", default="data/data/derived_masks", help="where to write masks")
    p.add_argument("--max-images", type=int, default=0, help="limit for quick test (0=all)")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--create-dirs", action="store_true", help="If set, the script will create missing output directories. Otherwise it will warn and skip saves.")
    args = p.parse_args()
    main(args)
