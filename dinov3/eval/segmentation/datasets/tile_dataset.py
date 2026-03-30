# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Dataset for loading satellite/aerial image tiles with segmentation masks.

Expected directory structure:
    root/
        train/
            images/    # .png, .jpg, .tif, .tiff
            masks/     # matching filenames, single-channel, pixel value = class index
        val/
            images/
            masks/
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _load_image(path: str) -> Image.Image:
    """Load an image, handling both standard formats and GeoTIFF."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(path) as src:
                # Read first 3 bands as RGB
                if src.count >= 3:
                    data = src.read([1, 2, 3])  # (3, H, W)
                else:
                    data = src.read(1)  # (H, W) single band
                    data = np.stack([data, data, data])  # make RGB
                # Transpose to (H, W, 3) and convert to uint8 if needed
                data = np.transpose(data, (1, 2, 0))
                if data.dtype != np.uint8:
                    # Normalize to 0-255
                    data = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
                return Image.fromarray(data)
        except ImportError:
            # Fallback to PIL for simple TIFF files
            return Image.open(path).convert("RGB")
    return Image.open(path).convert("RGB")


def _load_mask(path: str) -> Image.Image:
    """Load a single-channel mask."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(path) as src:
                data = src.read(1)  # (H, W) single band
                return Image.fromarray(data)
        except ImportError:
            return Image.open(path)
    return Image.open(path)


class TileSegmentationDataset(Dataset):
    """Dataset for satellite/aerial tile segmentation.

    Args:
        root: Path to split directory (e.g., root/train/)
        transforms: Optional transform applied to (image, mask) tuple
    """

    def __init__(self, root: str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "masks")

        assert os.path.isdir(self.images_dir), f"Images directory not found: {self.images_dir}"
        assert os.path.isdir(self.masks_dir), f"Masks directory not found: {self.masks_dir}"

        # Collect image files and match with masks
        self.samples = []
        for fname in sorted(os.listdir(self.images_dir)):
            stem = Path(fname).stem
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            # Find matching mask (try same extension first, then .png)
            mask_path = None
            for mask_ext in [ext, ".png", ".tif", ".tiff"]:
                candidate = os.path.join(self.masks_dir, stem + mask_ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is not None:
                self.samples.append((os.path.join(self.images_dir, fname), mask_path))

        assert len(self.samples) > 0, f"No image-mask pairs found in {root}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = _load_image(img_path)
        mask = _load_mask(mask_path)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask
