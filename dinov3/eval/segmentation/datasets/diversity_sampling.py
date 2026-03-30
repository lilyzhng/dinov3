# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Diversity sampling using DINOv3 features and k-Means clustering.

Selects a diverse subset of tiles from a large pool by:
1. Extracting DINOv3 CLS token features for each tile
2. Clustering with k-Means
3. Selecting the tile nearest to each cluster centroid

Usage:
    python -m dinov3.eval.segmentation.datasets.diversity_sampling \
        --tile_dir /path/to/tiles \
        --n_select 10000 \
        --output selected_tiles.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class TileImageDataset(Dataset):
    """Simple dataset for loading tile images for feature extraction."""

    def __init__(self, tile_dir: str, transform=None):
        self.tile_dir = tile_dir
        self.transform = transform
        self.files = sorted(
            f for f in os.listdir(tile_dir) if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        )
        assert len(self.files) > 0, f"No image files found in {tile_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.tile_dir, self.files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, idx


def extract_features(
    backbone: torch.nn.Module,
    tile_dir: str,
    batch_size: int = 32,
    device: str = "cuda",
    image_size: int = 518,
) -> tuple[np.ndarray, list[str]]:
    """Extract CLS token features from all tiles using a DINOv3 backbone.

    Args:
        backbone: DINOv3 backbone model (frozen).
        tile_dir: Directory containing tile images.
        batch_size: Batch size for feature extraction.
        device: Device to run inference on.
        image_size: Input image size for the backbone.

    Returns:
        features: (N, embed_dim) numpy array of CLS features.
        filenames: List of tile filenames.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TileImageDataset(tile_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    backbone = backbone.to(device)
    backbone.eval()

    all_features = []
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            # DINOv3 forward returns CLS token as first element
            output = backbone(images)
            if isinstance(output, dict):
                cls_features = output["x_norm_clstoken"]
            else:
                cls_features = output
            all_features.append(cls_features.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    return features, dataset.files


def kmeans_sampling(
    features: np.ndarray,
    n_select: int,
    seed: int = 42,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Select diverse samples using k-Means clustering.

    Args:
        features: (N, D) feature array.
        n_select: Number of samples to select.
        seed: Random seed for reproducibility.
        max_iter: Maximum k-Means iterations.

    Returns:
        selected_indices: (n_select,) array of selected sample indices.
        cluster_labels: (N,) array of cluster assignments for all samples.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_samples = features.shape[0]
    n_select = min(n_select, n_samples)

    logger.info(f"Running k-Means with K={n_select} on {n_samples} samples")

    kmeans = MiniBatchKMeans(
        n_clusters=n_select,
        random_state=seed,
        max_iter=max_iter,
        batch_size=min(10000, n_samples),
    )
    cluster_labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_

    # For each cluster, find the sample nearest to the centroid
    selected_indices = np.zeros(n_select, dtype=np.int64)
    for k in range(n_select):
        cluster_mask = cluster_labels == k
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            # Empty cluster — pick random sample
            selected_indices[k] = np.random.RandomState(seed + k).randint(0, n_samples)
            continue
        cluster_features = features[cluster_indices]
        distances = np.linalg.norm(cluster_features - centroids[k], axis=1)
        nearest = cluster_indices[np.argmin(distances)]
        selected_indices[k] = nearest

    return selected_indices, cluster_labels


def save_selection(
    filenames: list[str],
    selected_indices: np.ndarray,
    output_path: str,
):
    """Save selected tile filenames to JSON."""
    selected_files = [filenames[i] for i in selected_indices]
    result = {
        "n_total": len(filenames),
        "n_selected": len(selected_files),
        "selected_files": selected_files,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved {len(selected_files)} selected tiles to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Diversity sampling with DINOv3 features")
    parser.add_argument("--tile_dir", required=True, help="Directory of tile images")
    parser.add_argument("--n_select", type=int, required=True, help="Number of tiles to select")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--features_path", default=None, help="Path to pre-extracted features .npy file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.features_path and os.path.exists(args.features_path):
        logger.info(f"Loading pre-extracted features from {args.features_path}")
        data = np.load(args.features_path, allow_pickle=True).item()
        features = data["features"]
        filenames = data["filenames"]
    else:
        # Load backbone (requires DINOv3 to be installed)
        logger.info("Loading DINOv3 backbone for feature extraction")
        backbone = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl14")
        features, filenames = extract_features(backbone, args.tile_dir, batch_size=args.batch_size)

        # Optionally save features for reuse
        if args.features_path:
            np.save(args.features_path, {"features": features, "filenames": filenames})
            logger.info(f"Saved features to {args.features_path}")

    selected_indices, cluster_labels = kmeans_sampling(features, args.n_select, seed=args.seed)
    save_selection(filenames, selected_indices, args.output)


if __name__ == "__main__":
    main()
