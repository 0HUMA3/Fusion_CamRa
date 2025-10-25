import json, os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class RadarFusionDataset(Dataset):
    """
    Dataset for Radar–Camera–Depth fusion.
    Returns:
        {
            "image": Tensor [3, H, W],
            "dacc":  Tensor [1, H, W],
            "radar": Tensor [3, N],
            "mask":  Tensor [1, H, W] (zeros if not available),
            "meta":  dict
        }
    """
    def __init__(self, meta_json, use_processed=True, transform=None, radar_points=128):
        with open(meta_json) as f:
            self.meta = json.load(f)
        self.transform = transform
        self.use_processed = use_processed
        self.radar_points = radar_points

    def _choose_path(self, it):
        """Prefer processed_path if it exists; else fallback to original."""
        if self.use_processed and it.get("processed_path") and os.path.exists(it["processed_path"]):
            return it["processed_path"]
        if it.get("original_path") and os.path.exists(it["original_path"]):
            return it["original_path"]
        return it.get("processed_path") or it.get("original_path")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        it = self.meta[idx]

        # ------------------- Load Image -------------------
        img_path = self._choose_path(it)
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}  (meta idx {idx})")

        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img).astype(np.float32) / 255.0  # H,W,C
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # C,H,W

        # ------------------- Load Mask (optional) -------------------
        mask = None
        if it.get("mask_path"):
            mp = it["mask_path"]
            if os.path.exists(mp):
                mask = np.load(mp)
                mask = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)

        # ------------------- Load DACC (depth accumulation) -------------------
        dacc = None
        if it.get("dacc_path"):
            dp = it["dacc_path"]
            if os.path.exists(dp):
                try:
                    arr = np.load(dp)
                    if isinstance(arr, np.lib.npyio.NpzFile):
                        depth = arr.get("depth", None)
                        if depth is None:
                            depth = arr[arr.files[0]] if arr.files else None
                        if depth is not None:
                            dacc = torch.from_numpy(depth.astype(np.float32))
                    else:
                        dacc = torch.from_numpy(arr.astype(np.float32))
                except Exception as e:
                    print(f"[WARN] could not load dacc for {dp}: {e}")
                    dacc = None

        # Resize dacc to match image shape
        if dacc is not None:
            if dacc.ndim == 2:
                dacc = dacc.unsqueeze(0)
            H, W = img.shape[1:]
            dacc = F.interpolate(dacc.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        else:
            dacc = torch.zeros((1, *img.shape[1:]), dtype=torch.float32)

        # ------------------- Load Radar -------------------
        radar = torch.zeros((3, self.radar_points), dtype=torch.float32)
        rp = it.get("radar_path")
        if rp and os.path.exists(rp):
            try:
                arr = np.load(rp)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 3)
                N = self.radar_points
                if arr.shape[0] > N:
                    arr = arr[:N]
                elif arr.shape[0] < N:
                    pad = np.zeros((N - arr.shape[0], 3), dtype=np.float32)
                    arr = np.vstack([arr, pad])
                radar = torch.from_numpy(arr.T).contiguous()
            except Exception as e:
                print(f"[WARN] could not load radar for {rp}: {e}")

        # ------------------- Final Sample -------------------
        if mask is None:
            mask = torch.zeros((1, *img.shape[1:]), dtype=torch.float32)  # ✅ ensure consistent shape

        sample = {
            "image": img.float(),
            "dacc": dacc,
            "radar": radar,
            "mask": mask,   # ✅ always a tensor now
            "meta": it,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
