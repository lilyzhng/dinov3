# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Synthetic data generator for testing lane-level segmentation.

Generates synthetic satellite tile images with known lane line patterns
and their corresponding segmentation masks.
"""

import json
import math
import os
import tempfile

import cv2
import numpy as np
from PIL import Image


# Lane class indices (matching lane_classes.yaml)
BACKGROUND = 0
DOUBLE_SOLID = 1
WIDE_LANE = 2
ROAD_CURVATURE = 3
ROAD_DIVIDER = 4
INTERSECTION = 5
URBAN_LANE = 6
HIGHWAY_LANE = 7

CLASS_BUFFER = {
    DOUBLE_SOLID: 3,
    WIDE_LANE: 5,
    ROAD_CURVATURE: 3,
    ROAD_DIVIDER: 4,
    INTERSECTION: 0,
    URBAN_LANE: 3,
    HIGHWAY_LANE: 3,
}


def generate_straight_lines(size=512, n_lines=4, class_id=HIGHWAY_LANE):
    """Generate parallel straight horizontal lane lines."""
    img = np.random.randint(60, 120, (size, size, 3), dtype=np.uint8)  # road-like gray
    mask = np.zeros((size, size), dtype=np.uint8)

    spacing = size // (n_lines + 1)
    buffer = CLASS_BUFFER.get(class_id, 3)

    for i in range(1, n_lines + 1):
        y = i * spacing
        cv2.line(img, (0, y), (size - 1, y), (255, 255, 255), buffer)
        cv2.line(mask, (0, y), (size - 1, y), int(class_id), buffer)

    return img, mask


def generate_gentle_curve(size=512, class_id=ROAD_CURVATURE):
    """Generate a gentle curve (highway bend)."""
    img = np.random.randint(60, 120, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    buffer = CLASS_BUFFER.get(class_id, 3)

    # Sinusoidal curve with large wavelength
    points = []
    for x in range(0, size, 2):
        y = int(size // 2 + 80 * math.sin(2 * math.pi * x / size))
        points.append([x, y])
    points = np.array(points, dtype=np.int32)

    cv2.polylines(img, [points], False, (255, 255, 255), buffer)
    cv2.polylines(mask, [points], False, int(class_id), buffer)

    return img, mask


def generate_lane_merging(size=512):
    """Generate lane merging/splitting (on-ramp pattern)."""
    img = np.random.randint(60, 120, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Main lane (straight)
    y_main = size // 3
    cv2.line(img, (0, y_main), (size - 1, y_main), (255, 255, 255), 3)
    cv2.line(mask, (0, y_main), (size - 1, y_main), HIGHWAY_LANE, 3)

    # Merging lane (diagonal converging)
    y_start = 2 * size // 3
    points = []
    for x in range(0, size, 2):
        t = x / size
        y = int(y_start + (y_main - y_start) * t)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], False, (255, 255, 255), 3)
    cv2.polylines(mask, [points], False, HIGHWAY_LANE, 3)

    # Divider between them
    y_div = (y_main + y_start) // 2
    cv2.line(img, (0, y_div), (size // 2, y_div), (255, 200, 0), 4)
    cv2.line(mask, (0, y_div), (size // 2, y_div), ROAD_DIVIDER, 4)

    return img, mask


def generate_wide_divider(size=512):
    """Generate wide highway median divider."""
    img = np.random.randint(60, 120, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    y_center = size // 2
    cv2.line(img, (0, y_center), (size - 1, y_center), (255, 200, 0), 8)
    cv2.line(mask, (0, y_center), (size - 1, y_center), WIDE_LANE, 5)

    return img, mask


def generate_tight_curve(size=512, class_id=URBAN_LANE):
    """Generate tight curve (city turn)."""
    img = np.random.randint(60, 120, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    buffer = CLASS_BUFFER.get(class_id, 3)

    # Quarter-circle arc
    center = (size // 4, size // 4)
    radius = size // 3
    points = []
    for angle in range(0, 91, 2):
        rad = math.radians(angle)
        x = int(center[0] + radius * math.cos(rad))
        y = int(center[1] + radius * math.sin(rad))
        points.append([x, y])
    points = np.array(points, dtype=np.int32)

    cv2.polylines(img, [points], False, (255, 255, 255), buffer)
    cv2.polylines(mask, [points], False, int(class_id), buffer)

    return img, mask


def generate_four_way_intersection(size=512):
    """Generate a 4-way intersection."""
    img = np.random.randint(60, 120, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    center = size // 2

    # Draw intersection area as polygon
    margin = size // 6
    pts = np.array([
        [center - margin, center - margin],
        [center + margin, center - margin],
        [center + margin, center + margin],
        [center - margin, center + margin],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (100, 100, 100))
    cv2.fillPoly(mask, [pts], INTERSECTION)

    # Four approaching road lanes
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        if dx == 0:
            # Vertical approach
            y_start = center + dy * margin
            y_end = center + dy * (size // 2)
            cv2.line(img, (center - 20, y_start), (center - 20, y_end), (255, 255, 255), 3)
            cv2.line(mask, (center - 20, y_start), (center - 20, y_end), URBAN_LANE, 3)
            cv2.line(img, (center + 20, y_start), (center + 20, y_end), (255, 255, 255), 3)
            cv2.line(mask, (center + 20, y_start), (center + 20, y_end), URBAN_LANE, 3)
        else:
            # Horizontal approach
            x_start = center + dx * margin
            x_end = center + dx * (size // 2)
            cv2.line(img, (x_start, center - 20), (x_end, center - 20), (255, 255, 255), 3)
            cv2.line(mask, (x_start, center - 20), (x_end, center - 20), URBAN_LANE, 3)
            cv2.line(img, (x_start, center + 20), (x_end, center + 20), (255, 255, 255), 3)
            cv2.line(mask, (x_start, center + 20), (x_end, center + 20), URBAN_LANE, 3)

    # Double solid center lines
    cv2.line(img, (center, 0), (center, center - margin), (255, 255, 0), 3)
    cv2.line(mask, (center, 0), (center, center - margin), DOUBLE_SOLID, 3)
    cv2.line(img, (center, center + margin), (center, size - 1), (255, 255, 0), 3)
    cv2.line(mask, (center, center + margin), (center, size - 1), DOUBLE_SOLID, 3)

    return img, mask


def generate_t_junction(size=512):
    """Generate a T-junction intersection."""
    img = np.random.randint(60, 120, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    center_x, center_y = size // 2, size // 2

    # Horizontal road
    cv2.line(img, (0, center_y - 20), (size - 1, center_y - 20), (255, 255, 255), 3)
    cv2.line(mask, (0, center_y - 20), (size - 1, center_y - 20), URBAN_LANE, 3)
    cv2.line(img, (0, center_y + 20), (size - 1, center_y + 20), (255, 255, 255), 3)
    cv2.line(mask, (0, center_y + 20), (size - 1, center_y + 20), URBAN_LANE, 3)

    # Vertical road coming from bottom
    cv2.line(img, (center_x - 20, center_y + 20), (center_x - 20, size - 1), (255, 255, 255), 3)
    cv2.line(mask, (center_x - 20, center_y + 20), (center_x - 20, size - 1), URBAN_LANE, 3)
    cv2.line(img, (center_x + 20, center_y + 20), (center_x + 20, size - 1), (255, 255, 255), 3)
    cv2.line(mask, (center_x + 20, center_y + 20), (center_x + 20, size - 1), URBAN_LANE, 3)

    return img, mask


def generate_dense_multilane(size=512, n_lanes=6):
    """Generate dense parallel lanes (multi-lane urban road)."""
    img = np.random.randint(60, 120, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    spacing = size // (n_lanes + 1)
    for i in range(1, n_lanes + 1):
        y = i * spacing
        # Alternate between solid and dashed
        if i == n_lanes // 2:
            cv2.line(img, (0, y), (size - 1, y), (255, 255, 0), 3)
            cv2.line(mask, (0, y), (size - 1, y), DOUBLE_SOLID, 3)
        else:
            cv2.line(img, (0, y), (size - 1, y), (255, 255, 255), 3)
            cv2.line(mask, (0, y), (size - 1, y), URBAN_LANE, 3)

    return img, mask


# Scenario generators grouped by type
HIGHWAY_SCENARIOS = {
    "straight_lines": generate_straight_lines,
    "gentle_curve": generate_gentle_curve,
    "lane_merging": generate_lane_merging,
    "wide_divider": generate_wide_divider,
}

URBAN_SCENARIOS = {
    "tight_curve": generate_tight_curve,
    "four_way_intersection": generate_four_way_intersection,
    "t_junction": generate_t_junction,
    "dense_multilane": generate_dense_multilane,
}


def generate_synthetic_dataset(
    output_dir: str,
    n_per_scenario: int = 10,
    size: int = 512,
    scenarios: dict | None = None,
):
    """Generate a complete synthetic dataset.

    Args:
        output_dir: Root directory for the dataset.
        n_per_scenario: Number of tiles per scenario.
        size: Tile size in pixels.
        scenarios: Dict of scenario_name -> generator_fn. If None, uses all.

    Returns:
        List of (image_path, mask_path) tuples.
    """
    if scenarios is None:
        scenarios = {**HIGHWAY_SCENARIOS, **URBAN_SCENARIOS}

    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    pairs = []
    for name, gen_fn in scenarios.items():
        for i in range(n_per_scenario):
            img, mask_arr = gen_fn(size=size)

            fname = f"{name}_{i:04d}.png"
            img_path = os.path.join(images_dir, fname)
            mask_path = os.path.join(masks_dir, fname)

            Image.fromarray(img).save(img_path)
            Image.fromarray(mask_arr).save(mask_path)
            pairs.append((img_path, mask_path))

    return pairs


def generate_geojson_for_tile(
    tile_width: int = 512,
    tile_height: int = 512,
    n_lines: int = 3,
    class_names: list[str] | None = None,
) -> dict:
    """Generate a GeoJSON with synthetic polylines for testing rasterization.

    Returns a GeoJSON FeatureCollection with LineString features in pixel coords.
    """
    if class_names is None:
        class_names = ["highway_lane", "double_solid", "urban_lane"]

    features = []
    for i in range(n_lines):
        y = (i + 1) * tile_height // (n_lines + 1)
        coords = [[0, y], [tile_width // 4, y + 10], [tile_width // 2, y], [3 * tile_width // 4, y - 10], [tile_width - 1, y]]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "class": class_names[i % len(class_names)],
            },
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}
