# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Downsample tile images and masks for resolution ablation studies.

Given a dataset at native resolution (e.g. 7.5cm/px), produces a
downsampled copy (e.g. 15cm/px via 2× downsample). Images are resized
with LANCZOS (anti-aliased), masks with NEAREST (preserves class indices).

Usage:
    python -m dinov3.eval.segmentation.datasets.downsample_tiles \
        --input_dir /path/to/tiles_7.5cm \
        --output_dir /path/to/tiles_15cm \
        --scale 0.5
"""

import argparse
import logging
import os
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def downsample_directory(
    input_dir: str,
    output_dir: str,
    scale: float,
    image_resample=Image.LANCZOS,
    mask_resample=Image.NEAREST,
) -> int:
    """Downsample all images/masks in a directory.

    Args:
        input_dir: Source directory containing image files.
        output_dir: Destination directory (created if needed).
        scale: Scale factor (e.g. 0.5 for 2× downsample).
        image_resample: Resampling filter for images.
        mask_resample: Resampling filter for masks.

    Returns:
        Number of files processed.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for fname in sorted(os.listdir(input_dir)):
        if Path(fname).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        img = Image.open(src_path)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)

        # Use nearest-neighbor for single-channel masks, LANCZOS for RGB images
        if img.mode in ("L", "P") or (img.mode == "I" and img.getbands() == ("I",)):
            resized = img.resize((new_w, new_h), mask_resample)
        else:
            resized = img.resize((new_w, new_h), image_resample)

        resized.save(dst_path)
        count += 1

    return count


def downsample_split(input_root: str, output_root: str, scale: float):
    """Downsample a full split directory (images/ + masks/).

    Args:
        input_root: Source split directory (e.g. data/train/).
        output_root: Destination split directory.
        scale: Scale factor.
    """
    for subdir in ("images", "masks"):
        src = os.path.join(input_root, subdir)
        dst = os.path.join(output_root, subdir)
        if not os.path.isdir(src):
            logger.warning(f"Skipping missing directory: {src}")
            continue

        is_mask = subdir == "masks"
        if is_mask:
            count = downsample_directory(src, dst, scale, mask_resample=Image.NEAREST)
        else:
            count = downsample_directory(src, dst, scale)
        logger.info(f"Downsampled {count} files: {src} -> {dst} (scale={scale})")


def main():
    parser = argparse.ArgumentParser(description="Downsample tiles for resolution ablation")
    parser.add_argument("--input_dir", required=True, help="Input dataset root (with images/ and masks/ subdirs, or flat)")
    parser.add_argument("--output_dir", required=True, help="Output dataset root")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor (default: 0.5 = 2x downsample)")
    parser.add_argument("--flat", action="store_true", help="Treat input_dir as a flat image directory (no images/masks split)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.flat:
        count = downsample_directory(args.input_dir, args.output_dir, args.scale)
        logger.info(f"Downsampled {count} files (scale={args.scale})")
    else:
        downsample_split(args.input_dir, args.output_dir, args.scale)


if __name__ == "__main__":
    main()
