"""
NeuroCamo - COD10K Dataset Loader
===================================
This file loads the COD10K camouflaged object detection dataset from disk.

COD10K Dataset Structure (what we expect after download):
  e:\major_project\neurocamo\data\COD10K\
      Train\
          Image\       <- .jpg camouflaged images
          GT_Object\   <- .png binary masks (white = camouflaged object)
      Test\
          Image\
          GT_Object\

HOW IT WORKS:
  1. Scans the Image folder to find all .jpg files
  2. For each image, loads the matching binary mask from GT_Object folder
  3. Derives bounding box coordinates from the mask (finds the object region)
  4. Applies data augmentations (flip, blur, contrast)
  5. Normalizes pixel values to [0, 1] range for the Poisson Encoder

Reference: Fan et al. 2020 - COD10K Dataset
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class COD10KDataset(Dataset):
    """
    PyTorch Dataset for the COD10K camouflage detection benchmark.

    Args:
        root_dir   (str):  Path to COD10K folder. e.g. "e:/major_project/neurocamo/data/COD10K"
        split      (str):  Either "Train" or "Test"
        img_size   (int):  Resize all images to this square size. Default: 128
        augment    (bool): Apply data augmentation (only for Train split)
    """

    def __init__(self, root_dir, split: str = "Train",
                 img_size: int = 128, augment: bool = True,
                 return_mask: bool = False, mask_grid: int = 16):

        self.img_size = img_size
        self.augment  = augment and (split == "Train")  # Only augment during training
        # return_mask adds a (1, mask_grid, mask_grid) soft occupancy mask to the
        # sample. COD10K/NC4K ship pixel masks and the box-only pipeline threw
        # them away, leaving one supervised cell per image. Returning the mask
        # gives every one of the mask_grid^2 cells a label instead.
        self.return_mask = return_mask
        self.mask_grid = mask_grid

        # root_dir may be a single path or a list of paths. Passing a list lets
        # COD10K and NC4K be trained on together — both ship the identical
        # <root>/<split>/{Image,GT_Object} layout with .jpg images and
        # same-stem .png masks, so no per-dataset special casing is needed.
        roots = [root_dir] if isinstance(root_dir, (str, bytes, os.PathLike)) else list(root_dir)

        # (image_path, mask_path) pairs, so several roots can coexist even when
        # their file names collide.
        self.samples = []
        for root in roots:
            img_dir  = os.path.join(root, split, "Image")
            mask_dir = os.path.join(root, split, "GT_Object")
            if not os.path.isdir(img_dir):
                print(f"  [{split}] SKIP missing {img_dir}")
                continue
            names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
            for n in names:
                self.samples.append((
                    os.path.join(img_dir, n),
                    os.path.join(mask_dir, os.path.splitext(n)[0] + ".png"),
                ))
            print(f"  [{split}] Loaded {len(names)} images from {img_dir}")

        if len(roots) > 1:
            print(f"  [{split}] TOTAL {len(self.samples)} images across {len(roots)} datasets")

        # Kept for backwards compatibility with tools that read `.img_files`
        # and `.img_dir` (e.g. neurocamo-web/tools/diagnose_head.py).
        self.img_dir   = os.path.join(roots[0], split, "Image")
        self.mask_dir  = os.path.join(roots[0], split, "GT_Object")
        self.img_files = [os.path.basename(p) for p, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Loads one image + its mask, applies augmentations, returns tensors.

        Returns:
            image  (Tensor): Normalized image    shape (3, H, W), values in [0, 1]
            bbox   (Tensor): Bounding box        shape (4,)  -> [x_center, y_center, width, height]
                             Zero width/height means "no camouflaged object here"
            label  (Tensor): 1.0 if an object is present, 0.0 for a NonCAM background
        """
        img_path, mask_path = self.samples[idx]

        # ── 1. Load image (BGR) and convert to RGB ─────────────────────────
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Unreadable image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── 2. Load binary mask (white = object, black = background) ────────
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # No mask on disk == no annotated object; treat as a background frame
            # rather than silently crashing mid-epoch.
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # ── 3. Resize both to target size ────────────────────────────────────
        image = cv2.resize(image, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)

        # ── 4. Data Augmentation (only during training) ───────────────────────
        if self.augment:
            image, mask = self._augment(image, mask)

        # ── 5. Derive bounding box from mask ─────────────────────────────────
        bbox = self._mask_to_bbox(mask)

        # ── 6. Normalize image pixels to [0.0, 1.0] for Poisson Encoder ──────
        image = image.astype(np.float32) / 255.0  

        # ── 7. Convert to PyTorch tensors ─────────────────────────────────────
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
        bbox_tensor  = torch.tensor(bbox, dtype=torch.float32)
        has_object   = float(bbox[2] > 0.01 and bbox[3] > 0.01)
        label_tensor = torch.tensor([has_object], dtype=torch.float32)

        if not self.return_mask:
            return image_tensor, bbox_tensor, label_tensor

        # Soft occupancy per cell: area-average the binary mask down to the
        # detector grid, so a cell half-covered by the object gets 0.5 rather
        # than being forced to 0 or 1 by nearest-neighbour sampling.
        g = self.mask_grid
        small = cv2.resize((mask > 127).astype(np.float32), (g, g),
                           interpolation=cv2.INTER_AREA)
        mask_tensor = torch.from_numpy(small).unsqueeze(0)   # (1, g, g)

        return image_tensor, bbox_tensor, label_tensor, mask_tensor

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        """Apply random augmentations to an image and its mask."""

        # Random horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask  = cv2.flip(mask,  1)

        # Random Gaussian blur to simulate atmospheric haze (30% chance)
        if np.random.rand() > 0.7:
            ksize = np.random.choice([3, 5])           # kernel size (must be odd)
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # Random brightness/contrast jitter (50% chance)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)        # contrast factor
            beta  = np.random.randint(-20, 20)          # brightness offset
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        return image, mask

    def _mask_to_bbox(self, mask: np.ndarray):
        """
        Derives a bounding box from a binary mask.
        Returns [x_center, y_center, width, height] normalized to [0, 1].

        An empty mask means there is genuinely no camouflaged object in the
        frame — 2960 of COD10K's 6000 training images are 'NonCAM' backgrounds
        with an all-black mask. It MUST return a zero-size box so that
        `has_object` in __getitem__ evaluates to 0.0.

        A previous version returned [0.5, 0.5, 0.1, 0.1] here, which made
        `has_object` come out as 1.0. That taught the model, on half of every
        epoch, that "an object is present, dead centre, 10% of the frame,
        confidence 1.0". That single line is what produced the collapsed
        detector: centre pinned to (0.50, 0.50) with std 0.005, and objectness
        saturated at ~0.998 on literally any input.
        """
        # Find pixels belonging to the object (value > 127 = white = object)
        rows = np.any(mask > 127, axis=1)
        cols = np.any(mask > 127, axis=0)

        # No object region -> null box, so has_object becomes 0.0
        if not rows.any():
            return [0.5, 0.5, 0.0, 0.0]

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        H, W = mask.shape

        # Convert to center format and normalize to [0, 1]
        x_center = float(np.clip((x_min + x_max) / 2.0 / W, 0.01, 0.99))
        y_center = float(np.clip((y_min + y_max) / 2.0 / H, 0.01, 0.99))
        width    = float(np.clip(max(0.01, (x_max - x_min) / W), 0.01, 0.99))
        height   = float(np.clip(max(0.01, (y_max - y_min) / H), 0.01, 0.99))

        return [x_center, y_center, width, height]


def get_dataloaders(data_root: str, img_size: int = 128, batch_size: int = 8, include_negatives: bool = True):
    """
    Creates training and validation DataLoader objects.
    Mixes negative (non-camo) images into the training set if available.
    """
    train_dataset = COD10KDataset(data_root, split="Train", img_size=img_size, augment=True)
    val_dataset   = COD10KDataset(data_root, split="Test",  img_size=img_size, augment=False)

    if include_negatives:
        neg_dir = os.path.join(os.path.dirname(data_root), "negatives")
        if os.path.exists(neg_dir) and len([f for f in os.listdir(neg_dir) if f.endswith(('.jpg', '.png'))]) > 0:
            from datasets.negative_dataset import NegativeDataset
            from torch.utils.data import ConcatDataset
            neg_dataset = NegativeDataset(neg_dir=neg_dir, img_size=img_size)
            train_dataset = ConcatDataset([train_dataset, neg_dataset])
            print(f"  [CombinedDataset] Mixed {len(neg_dataset)} negative samples into training set.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick structure test (runs even without the dataset downloaded yet)
    print("=" * 55)
    print("  NeuroCamo - DataLoader Structure Test")
    print("=" * 55)
    print()

    DATA_ROOT = r"e:\major_project\neurocamo\data\COD10K"

    if not os.path.exists(os.path.join(DATA_ROOT, "Train", "Image")):
        print("  ⚠️  COD10K dataset not found at:")
        print(f"     {DATA_ROOT}")
        print()
        print("  Please download it using the instructions in:")
        print("  e:\\major_project\\neurocamo\\data\\DOWNLOAD_INSTRUCTIONS.txt")
    else:
        print("  ✅ Dataset found! Loading...")
        train_loader, val_loader = get_dataloaders(DATA_ROOT)

        # Grab one batch and print its shape
        images, bboxes, labels = next(iter(train_loader))
        print(f"  Batch image shape : {list(images.shape)}")
        print(f"  Batch bbox shape  : {list(bboxes.shape)}")
        print(f"  Batch label shape : {list(labels.shape)}")
        print()
        print("  ✅ DataLoader working correctly!")

    print("=" * 55)
