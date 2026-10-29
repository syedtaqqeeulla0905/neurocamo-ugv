"""
NC4K Parquet Dataset Extractor
===============================
Extracts Hugging Face NC4K parquet dataset into standard image + GT mask folders
compatible with the NeuroCamo PyTorch DataLoader.

Destination: e:\\major_project\\neurocamo\\data\\NC4K_extracted\\
"""

import os
import io
import glob
import pandas as pd
from PIL import Image

NC4K_DIR = r"e:\major_project\neurocamo\data\NC4K\data"
OUT_DIR  = r"e:\major_project\neurocamo\data\NC4K_extracted"

os.makedirs(os.path.join(OUT_DIR, "Train", "Image"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "Train", "GT_Object"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "Test", "Image"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "Test", "GT_Object"), exist_ok=True)

parquet_files = glob.glob(os.path.join(NC4K_DIR, "*.parquet"))
print(f"Found {len(parquet_files)} NC4K parquet files.")

total_extracted = 0

for pfile in parquet_files:
    fname = os.path.basename(pfile)
    split = "Test" if ("test" in fname or "valid" in fname) else "Train"
    print(f"\nProcessing {fname} -> Split: {split}...")

    df = pd.read_parquet(pfile)
    for idx, row in df.iterrows():
        img_bytes = row["image"]["bytes"]
        gt_bytes  = row["gt"]["bytes"]

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        gt  = Image.open(io.BytesIO(gt_bytes)).convert("L")

        item_id = f"nc4k_{total_extracted:05d}"
        img_path = os.path.join(OUT_DIR, split, "Image", f"{item_id}.jpg")
        gt_path  = os.path.join(OUT_DIR, split, "GT_Object", f"{item_id}.png")

        img.save(img_path, "JPEG")
        gt.save(gt_path, "PNG")

        total_extracted += 1

print(f"\n✅ Extracted {total_extracted} NC4K image and mask pairs to {OUT_DIR}")
