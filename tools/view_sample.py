# tools/view_sample.py
import glob, os, sys
from PIL import Image
import matplotlib.pyplot as plt

orig_pattern = "data/data/original_data/**/**/*.jpg"
preproc_pattern = "data/data/preprocessed_data/**/**/*.jpg"

orig_files = sorted(glob.glob(orig_pattern, recursive=True))
preproc_files = sorted(glob.glob(preproc_pattern, recursive=True))

print(f"Found {len(orig_files)} original images")
print(f"Found {len(preproc_files)} preprocessed images")

if not orig_files:
    print("No original images found. Check path:", orig_pattern); sys.exit(1)
if not preproc_files:
    print("No preprocessed images found. Check path:", preproc_pattern); sys.exit(1)

orig = Image.open(orig_files[0])
pre = Image.open(preproc_files[0])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(orig)
axes[0].set_title(os.path.basename(orig_files[0]))
axes[0].axis('off')

axes[1].imshow(pre)
axes[1].set_title(os.path.basename(preproc_files[0]))
axes[1].axis('off')

plt.show()
