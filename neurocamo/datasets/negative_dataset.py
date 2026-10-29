"""
NeuroCamo - Negative Sample Dataset
=====================================
Downloads ~300 non-camouflage images from COCO val2017 (annotations-free,
just raw images) to teach the model what "no target" looks like.

Each negative sample:
  - Image:      any regular photo (person, car, room, food, etc.)
  - bbox_target: [0.5, 0.5, 0.0, 0.0]  (zero-size box, center of image)
  - conf_target: 0.0                    (NO object present)
"""

import os
import json
import random
import requests
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

# ── Config ────────────────────────────────────────────────────────────────────
COCO_VAL_JSON  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"
NEG_DIR        = r"e:\major_project\neurocamo\data\negatives"
TARGET_COUNT   = 300   # How many negative images to download

COCO_VAL_IDS = [
    # Pre-selected 300 COCO val2017 image IDs (common objects, zero camo)
    391895, 522418, 184613, 318219, 554625, 574769, 60623,  309022, 5802,
    222564, 118113, 193271, 224736, 403013, 374628, 328757, 384213, 86408,
    473406, 242611, 425226, 108099, 51191,  82839,  203317, 476415, 544852,
    262145, 285994, 459153, 196141, 559674, 462415, 17627,  430961, 458054,
    289393, 439715, 503461, 144941, 360661, 209222, 381480, 453584, 497020,
    249315, 341529, 530016, 113588, 337055, 410650, 234932, 386164, 354533,
    430023, 134886, 245357, 278705, 349860, 403385, 219283, 500478, 264363,
    568402, 193252, 475184, 78485,  374546, 397133, 386781, 213547, 118035,
    312421, 289393, 174482, 378176, 514508, 258985, 150417, 218491, 309391,
    289393, 484248, 219578, 283117, 164363, 445248, 262969, 179765, 356347,
    310391, 205282, 411530, 127135, 507249, 167126, 327696, 556543, 483050,
    388695, 269314, 536264, 487473, 488767, 537944, 353995, 262895, 459887,
    471241, 302860, 458444, 296284, 313294, 165284, 289553, 306344, 524894,
    403384, 229441, 156832, 568436, 434230, 185250, 253424, 494869, 263796,
    471917, 386296, 255664, 228842, 316050, 268290, 177196, 373051, 200365,
    571530, 321333, 252776, 224044, 326537, 186890, 503657, 282724, 240767,
    434818, 374399, 408251, 384004, 441586, 259021, 462572, 426773, 244773,
    427407, 393225, 449996, 391070, 313294, 218469, 361551, 384213, 176040,
    548012, 340697, 484790, 139099, 309813, 562121, 165352, 256668, 461197,
    239274, 252219, 289832, 308490, 196869, 178028, 503657, 439714, 382030,
    285994, 507373, 421699, 291490, 302343, 399831, 417779, 415978, 295441,
    411018, 340697, 425226, 460106, 399849, 239274, 165599, 477398, 562840,
    184791, 422178, 353884, 472750, 168427, 398473, 268890, 479185, 290003,
    484758, 388025, 484735, 229813, 391895, 265285, 278705, 314709, 322141,
    468208, 467154, 284790, 404069, 451300, 465022, 315173, 261982, 452649,
    443303, 497020, 553220, 338479, 459977, 443537, 329295, 453841, 428079,
    567998, 358316, 540186, 382335, 382591, 349860, 455157, 482836, 286994,
    474078, 509405, 481480, 426148, 369771, 522713, 468208, 400534, 326537,
    459520, 373320, 453841, 562150, 288454, 454750, 497736, 449996, 558840,
    488816, 406059, 350963, 345434, 536041, 452249, 469666, 289175, 470513,
    541773, 386781, 421378, 388275, 313294, 309813, 316659, 265285, 499887,
    571196, 434230, 492077, 396581, 507688, 313640, 392669, 545773, 467388,
    338479, 248063, 403384, 442480, 315003, 415645,
]


def download_negatives(target_dir: str = NEG_DIR, count: int = TARGET_COUNT):
    """Download COCO val images as negative samples."""
    os.makedirs(target_dir, exist_ok=True)

    ids = list(set(COCO_VAL_IDS))[:count]
    downloaded = 0
    skipped    = 0

    print(f"Downloading {len(ids)} negative images to {target_dir} ...")
    for img_id in ids:
        fname = os.path.join(target_dir, f"{img_id:012d}.jpg")
        if os.path.exists(fname):
            skipped += 1
            continue
        url = COCO_IMAGE_URL.format(img_id)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(fname, "wb") as f:
                    f.write(r.content)
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"  ... {downloaded} downloaded so far")
        except Exception as e:
            print(f"  Failed {img_id}: {e}")

    total = len(os.listdir(target_dir))
    print(f"Done. Downloaded: {downloaded} | Skipped (existing): {skipped} | Total on disk: {total}")
    return total


class NegativeDataset(Dataset):
    """
    Dataset of non-camouflage images.
    Labels: conf=0.0 (no object), bbox=[0.5,0.5,0,0] (null box)

    IMPORTANT: negatives must be preprocessed EXACTLY like COD10K positives —
    resize, then plain /255.0 into [0, 1] for the Poisson encoder. An earlier
    version ran ImageNet Normalize() and then clamped back to [0, 1], which
    crushed every negative into a different intensity distribution than the
    positives. The Poisson encoder turns intensity straight into firing rate, so
    the model could tell positive from negative by histogram alone and never had
    to learn anything about camouflage. That is why objectness saturated at ~1.0
    on real photos.
    """

    def __init__(self, neg_dir: str = NEG_DIR, img_size: int = 128, augment: bool = True):
        self.img_size = img_size
        self.augment = augment
        self.files = [
            os.path.join(neg_dir, f)
            for f in os.listdir(neg_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not self.files:
            raise FileNotFoundError(
                f"No negative images found in {neg_dir}. "
                "Run download_negatives() first."
            )
        print(f"[NegativeDataset] {len(self.files)} negative images loaded.")
        if len(self.files) < 500:
            print(f"[NegativeDataset] WARNING: only {len(self.files)} negatives. "
                  "Oversampling this few to balance thousands of positives makes "
                  "the objectness branch memorise them instead of generalising.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 200
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Same augmentations the positive loader uses, so the two streams stay
        # statistically indistinguishable apart from their content.
        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            if np.random.rand() > 0.7:
                k = int(np.random.choice([3, 5]))
                img = cv2.GaussianBlur(img, (k, k), 0)
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.randint(-20, 20)
                img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Targets: zero confidence, null bounding box
        bbox_target = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32)
        conf_target = torch.tensor([0.0], dtype=torch.float32)

        return tensor, bbox_target, conf_target


if __name__ == "__main__":
    print("Downloading negative samples...")
    n = download_negatives()
    print(f"\nTotal negatives on disk: {n}")

    if n > 0:
        ds = NegativeDataset()
        img, bbox, conf = ds[0]
        print(f"Sample shape: {img.shape}, bbox: {bbox}, conf: {conf}")
        print("NegativeDataset OK!")
