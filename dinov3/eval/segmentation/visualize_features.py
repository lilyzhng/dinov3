# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Visualize DINOv3 patch features via PCA projection.

Extracts patch-level features from a DINOv3 backbone, projects to 3
principal components, and saves RGB PCA visualizations. Useful for
comparing what the model "sees" across resolutions (e.g. 7.5cm vs 15cm).

Usage:
    python -m dinov3.eval.segmentation.visualize_features \
        --image_dir /path/to/tiles \
        --output_dir /path/to/vis \
        --model_name dinov3_vitl14

    # Compare two resolutions side-by-side:
    python -m dinov3.eval.segmentation.visualize_features \
        --image_dir /path/to/tiles_7.5cm \
        --image_dir_b /path/to/tiles_15cm \
        --output_dir /path/to/vis_comparison \
        --model_name dinov3_vitl14
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_backbone(model_name: str, device: str = "cuda"):
    """Load a DINOv3 backbone from torch hub."""
    model = torch.hub.load("facebookresearch/dinov3", model_name)
    model = model.to(device)
    model.eval()
    return model


def get_patch_features(
    backbone: torch.nn.Module,
    image: Image.Image,
    image_size: int = 518,
    device: str = "cuda",
) -> np.ndarray:
    """Extract patch-level features from a single image.

    Returns:
        features: (H_patches, W_patches, embed_dim) numpy array.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        # get_intermediate_layers returns list of (patch_tokens, cls_token)
        features = backbone.get_intermediate_layers(
            img_tensor, n=[backbone.n_blocks - 1], reshape=True, return_class_token=False
        )
        # features is a list with one element: (B, C, H_p, W_p)
        feat = features[0]  # (1, C, H_p, W_p)

    feat = feat.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H_p, W_p, C)
    return feat


def pca_project(features: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Project features to n_components via PCA.

    Args:
        features: (H, W, D) array.

    Returns:
        projected: (H, W, n_components) array, min-max normalized to [0, 255].
    """
    h, w, d = features.shape
    flat = features.reshape(-1, d).astype(np.float32)

    # Center
    mean = flat.mean(axis=0)
    flat_centered = flat - mean

    # PCA via SVD (faster than sklearn for small spatial dims)
    _, _, Vt = np.linalg.svd(flat_centered, full_matrices=False)
    components = Vt[:n_components]  # (n_components, D)

    projected = flat_centered @ components.T  # (H*W, n_components)
    projected = projected.reshape(h, w, n_components)

    # Normalize each component to [0, 255]
    for c in range(n_components):
        ch = projected[:, :, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            projected[:, :, c] = (ch - ch_min) / (ch_max - ch_min) * 255
        else:
            projected[:, :, c] = 0

    return projected.astype(np.uint8)


def visualize_single(
    backbone: torch.nn.Module,
    image_path: str,
    output_path: str,
    image_size: int = 518,
    device: str = "cuda",
    upscale: bool = True,
) -> np.ndarray:
    """Extract features, PCA-project, and save visualization for one image.

    Returns:
        pca_rgb: (H_p, W_p, 3) PCA visualization array.
    """
    img = Image.open(image_path).convert("RGB")
    features = get_patch_features(backbone, img, image_size=image_size, device=device)
    pca_rgb = pca_project(features, n_components=3)

    vis = Image.fromarray(pca_rgb)
    if upscale:
        vis = vis.resize((img.width, img.height), Image.BILINEAR)

    vis.save(output_path)
    return pca_rgb


def visualize_comparison(
    backbone: torch.nn.Module,
    image_path_a: str,
    image_path_b: str,
    output_path: str,
    image_size: int = 518,
    device: str = "cuda",
):
    """Create side-by-side PCA comparison of two images (e.g. different resolutions).

    Fits PCA jointly so colors are comparable.
    """
    img_a = Image.open(image_path_a).convert("RGB")
    img_b = Image.open(image_path_b).convert("RGB")

    feat_a = get_patch_features(backbone, img_a, image_size=image_size, device=device)
    feat_b = get_patch_features(backbone, img_b, image_size=image_size, device=device)

    # Joint PCA so colors are consistent
    h_a, w_a, d = feat_a.shape
    h_b, w_b, _ = feat_b.shape
    all_flat = np.concatenate([feat_a.reshape(-1, d), feat_b.reshape(-1, d)], axis=0).astype(np.float32)

    mean = all_flat.mean(axis=0)
    centered = all_flat - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:3]

    projected = centered @ components.T

    n_a = h_a * w_a
    proj_a = projected[:n_a].reshape(h_a, w_a, 3)
    proj_b = projected[n_a:].reshape(h_b, w_b, 3)

    # Joint normalization
    for c in range(3):
        ch_min = min(proj_a[:, :, c].min(), proj_b[:, :, c].min())
        ch_max = max(proj_a[:, :, c].max(), proj_b[:, :, c].max())
        if ch_max - ch_min > 1e-8:
            proj_a[:, :, c] = (proj_a[:, :, c] - ch_min) / (ch_max - ch_min) * 255
            proj_b[:, :, c] = (proj_b[:, :, c] - ch_min) / (ch_max - ch_min) * 255
        else:
            proj_a[:, :, c] = 0
            proj_b[:, :, c] = 0

    vis_a = Image.fromarray(proj_a.astype(np.uint8)).resize((img_a.width, img_a.height), Image.BILINEAR)
    vis_b = Image.fromarray(proj_b.astype(np.uint8)).resize((img_a.width, img_a.height), Image.BILINEAR)

    # Side-by-side: original_a | pca_a | pca_b | original_b
    canvas_w = img_a.width * 4
    canvas_h = img_a.height
    canvas = Image.new("RGB", (canvas_w, canvas_h))
    canvas.paste(img_a.resize((img_a.width, img_a.height)), (0, 0))
    canvas.paste(vis_a, (img_a.width, 0))
    canvas.paste(vis_b, (img_a.width * 2, 0))
    img_b_resized = img_b.resize((img_a.width, img_a.height), Image.BILINEAR)
    canvas.paste(img_b_resized, (img_a.width * 3, 0))
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features via PCA")
    parser.add_argument("--image_dir", required=True, help="Directory of input tiles (resolution A)")
    parser.add_argument("--image_dir_b", default=None, help="Optional second directory (resolution B) for comparison")
    parser.add_argument("--output_dir", required=True, help="Output directory for visualizations")
    parser.add_argument("--model_name", default="dinov3_vitl14", help="DINOv3 model name for torch.hub")
    parser.add_argument("--image_size", type=int, default=518, help="Input size for backbone")
    parser.add_argument("--max_images", type=int, default=50, help="Max number of images to visualize")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading backbone: {args.model_name}")
    backbone = load_backbone(args.model_name, device=args.device)

    image_files = sorted(
        f for f in os.listdir(args.image_dir) if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
    )[:args.max_images]

    for i, fname in enumerate(image_files):
        path_a = os.path.join(args.image_dir, fname)
        stem = Path(fname).stem

        if args.image_dir_b is not None:
            path_b = os.path.join(args.image_dir_b, fname)
            if not os.path.exists(path_b):
                logger.warning(f"No matching file for {fname} in {args.image_dir_b}, skipping")
                continue
            out_path = os.path.join(args.output_dir, f"{stem}_comparison.png")
            visualize_comparison(backbone, path_a, path_b, out_path, args.image_size, args.device)
        else:
            out_path = os.path.join(args.output_dir, f"{stem}_pca.png")
            visualize_single(backbone, path_a, out_path, args.image_size, args.device)

        logger.info(f"[{i + 1}/{len(image_files)}] Saved: {out_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
