# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Rasterize GeoJSON polyline annotations into segmentation masks.

Converts polyline labels (e.g., lane lines drawn in a labeling tool) into
pixel-level segmentation masks suitable for training.

Usage:
    python -m dinov3.eval.segmentation.datasets.rasterize \
        --geojson /path/to/annotations.geojson \
        --tiles /path/to/tiles/ \
        --output /path/to/masks/ \
        --classes /path/to/lane_classes.yaml
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_class_config(config_path: str) -> dict:
    """Load class configuration from YAML.

    Returns dict mapping class_name -> {index, buffer_px}.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["classes"]


def load_geojson(geojson_path: str) -> list[dict]:
    """Load GeoJSON features."""
    with open(geojson_path) as f:
        data = json.load(f)
    return data.get("features", [])


def rasterize_polylines(
    features: list[dict],
    tile_width: int,
    tile_height: int,
    class_config: dict,
    tile_bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Rasterize GeoJSON polyline features into a segmentation mask.

    Args:
        features: GeoJSON features with LineString/MultiLineString geometries.
            Each feature must have a "class" property matching a key in class_config.
        tile_width: Output mask width in pixels.
        tile_height: Output mask height in pixels.
        class_config: Dict mapping class_name -> {index, buffer_px}.
        tile_bounds: (min_x, min_y, max_x, max_y) geographic bounds of the tile.
            If None, coordinates are treated as pixel coordinates directly.

    Returns:
        mask: (H, W) uint8 array where each pixel = class index, 0 = background.
    """
    mask = np.zeros((tile_height, tile_width), dtype=np.uint8)

    for feature in features:
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        class_name = props.get("class", "")

        if class_name not in class_config:
            logger.warning(f"Unknown class '{class_name}', skipping feature")
            continue

        cls_info = class_config[class_name]
        class_index = cls_info["index"]
        buffer_px = cls_info.get("buffer_px", 3)

        if class_index == 0:
            continue  # background, skip

        geom_type = geom.get("type", "")
        if geom_type == "LineString":
            coords_list = [geom["coordinates"]]
        elif geom_type == "MultiLineString":
            coords_list = geom["coordinates"]
        elif geom_type == "Polygon":
            coords_list = geom["coordinates"]  # outer ring + holes
        else:
            logger.warning(f"Unsupported geometry type '{geom_type}', skipping")
            continue

        for coords in coords_list:
            points = _geo_to_pixel(coords, tile_width, tile_height, tile_bounds)
            points = points.astype(np.int32)

            if geom_type == "Polygon":
                cv2.fillPoly(mask, [points], color=int(class_index))
            else:
                if buffer_px > 0:
                    cv2.polylines(mask, [points], isClosed=False, color=int(class_index), thickness=buffer_px)
                else:
                    cv2.polylines(mask, [points], isClosed=False, color=int(class_index), thickness=1)

    return mask


def _geo_to_pixel(
    coords: list,
    tile_width: int,
    tile_height: int,
    tile_bounds: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Convert geographic coordinates to pixel coordinates.

    If tile_bounds is None, coords are assumed to already be in pixel space.
    """
    points = np.array(coords, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    # Take only x, y (ignore z if present)
    points = points[:, :2]

    if tile_bounds is not None:
        min_x, min_y, max_x, max_y = tile_bounds
        x_scale = tile_width / (max_x - min_x) if max_x != min_x else 1.0
        y_scale = tile_height / (max_y - min_y) if max_y != min_y else 1.0
        points[:, 0] = (points[:, 0] - min_x) * x_scale
        points[:, 1] = (max_y - points[:, 1]) * y_scale  # flip y axis (geo coords are bottom-up)

    return points


def rasterize_tiles(
    geojson_path: str,
    tiles_dir: str,
    output_dir: str,
    class_config_path: str,
    tile_size: tuple[int, int] | None = None,
):
    """Rasterize polylines for all tiles in a directory.

    Args:
        geojson_path: Path to GeoJSON file with polyline annotations.
        tiles_dir: Directory containing tile images.
        output_dir: Directory to write mask images.
        class_config_path: Path to class configuration YAML.
        tile_size: (width, height) override. If None, read from each tile image.
    """
    class_config = load_class_config(class_config_path)
    features = load_geojson(geojson_path)
    os.makedirs(output_dir, exist_ok=True)

    tile_files = sorted(
        f for f in os.listdir(tiles_dir) if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    logger.info(f"Rasterizing {len(features)} features across {len(tile_files)} tiles")

    for tile_file in tile_files:
        tile_path = os.path.join(tiles_dir, tile_file)

        if tile_size is not None:
            w, h = tile_size
        else:
            from PIL import Image

            with Image.open(tile_path) as img:
                w, h = img.size

        # For now, treat all features as belonging to this tile
        # (in practice, you'd filter by spatial intersection with tile bounds)
        mask = rasterize_polylines(features, w, h, class_config, tile_bounds=None)

        mask_filename = Path(tile_file).stem + ".png"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)

    logger.info(f"Wrote {len(tile_files)} masks to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Rasterize GeoJSON polylines to segmentation masks")
    parser.add_argument("--geojson", required=True, help="Path to GeoJSON annotations")
    parser.add_argument("--tiles", required=True, help="Directory of tile images")
    parser.add_argument("--output", required=True, help="Output directory for masks")
    parser.add_argument("--classes", required=True, help="Path to lane_classes.yaml")
    parser.add_argument("--tile-width", type=int, default=None, help="Override tile width")
    parser.add_argument("--tile-height", type=int, default=None, help="Override tile height")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tile_size = None
    if args.tile_width and args.tile_height:
        tile_size = (args.tile_width, args.tile_height)

    rasterize_tiles(args.geojson, args.tiles, args.output, args.classes, tile_size)


if __name__ == "__main__":
    main()
