# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Unit tests for polyline rasterization."""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from dinov3.eval.segmentation.datasets.rasterize import (
    load_class_config,
    rasterize_polylines,
)


def _make_class_config():
    return {
        "background": {"index": 0, "buffer_px": 0},
        "highway_lane": {"index": 7, "buffer_px": 3},
        "double_solid": {"index": 1, "buffer_px": 3},
        "urban_lane": {"index": 6, "buffer_px": 3},
        "wide_lane": {"index": 2, "buffer_px": 5},
    }


def _make_line_feature(coords, class_name):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"class": class_name},
    }


class TestRasterizeStraightLine:
    """Test rasterization of straight lines."""

    def test_horizontal_line(self):
        features = [_make_line_feature([[0, 128], [511, 128]], "highway_lane")]
        config = _make_class_config()
        mask = rasterize_polylines(features, 512, 256, config)

        assert mask.shape == (256, 512)
        assert mask[128, 256] == 7  # center pixel should be class 7
        assert mask[0, 0] == 0  # corner should be background

    def test_vertical_line(self):
        features = [_make_line_feature([[256, 0], [256, 511]], "double_solid")]
        config = _make_class_config()
        mask = rasterize_polylines(features, 512, 512, config)

        assert mask[256, 256] == 1

    def test_line_pixel_coverage(self):
        """Verify that a thick line covers the expected number of pixels."""
        features = [_make_line_feature([[0, 256], [511, 256]], "wide_lane")]
        config = _make_class_config()
        mask = rasterize_polylines(features, 512, 512, config)

        # Wide lane has buffer_px=5, so roughly 5 * 512 pixels
        nonzero = np.count_nonzero(mask)
        assert nonzero > 512 * 3  # at least 3px wide over 512 columns
        assert nonzero < 512 * 8  # but not too wide


class TestRasterizeCurves:
    """Test rasterization of curved lines."""

    def test_arc_has_nonzero_pixels(self):
        """A curved line should produce non-zero pixels."""
        import math

        coords = []
        for angle in range(0, 181, 5):
            rad = math.radians(angle)
            x = int(256 + 100 * math.cos(rad))
            y = int(256 + 100 * math.sin(rad))
            coords.append([x, y])

        features = [_make_line_feature(coords, "highway_lane")]
        config = _make_class_config()
        mask = rasterize_polylines(features, 512, 512, config)

        assert np.count_nonzero(mask) > 100  # arc should cover significant pixels


class TestRasterizeMultiClass:
    """Test multi-class rasterization."""

    def test_two_classes_no_overlap(self):
        features = [
            _make_line_feature([[0, 100], [511, 100]], "highway_lane"),
            _make_line_feature([[0, 400], [511, 400]], "double_solid"),
        ]
        config = _make_class_config()
        mask = rasterize_polylines(features, 512, 512, config)

        assert mask[100, 256] == 7  # highway_lane
        assert mask[400, 256] == 1  # double_solid
        assert mask[250, 256] == 0  # gap between them = background

    def test_overlapping_classes_last_wins(self):
        """When two lines overlap, the last one drawn wins."""
        features = [
            _make_line_feature([[0, 256], [511, 256]], "highway_lane"),
            _make_line_feature([[0, 256], [511, 256]], "double_solid"),
        ]
        config = _make_class_config()
        mask = rasterize_polylines(features, 512, 512, config)

        # Last class drawn should win at the center
        assert mask[256, 256] == 1  # double_solid was drawn last


class TestRasterizeBufferWidths:
    """Test different buffer widths."""

    @pytest.mark.parametrize("buffer_px,class_name", [(1, "highway_lane"), (3, "double_solid"), (5, "wide_lane")])
    def test_buffer_width_affects_coverage(self, buffer_px, class_name):
        features = [_make_line_feature([[0, 256], [511, 256]], class_name)]
        config = _make_class_config()
        # Override buffer to specific value for testing
        config[class_name]["buffer_px"] = buffer_px
        mask = rasterize_polylines(features, 512, 512, config)

        nonzero = np.count_nonzero(mask)
        # More buffer = more pixels
        assert nonzero >= 512 * max(1, buffer_px - 1)


class TestRasterizeEdgeCases:
    """Test edge cases."""

    def test_empty_features(self):
        mask = rasterize_polylines([], 512, 512, _make_class_config())
        assert mask.shape == (512, 512)
        assert np.all(mask == 0)

    def test_unknown_class_ignored(self):
        features = [_make_line_feature([[0, 256], [511, 256]], "nonexistent_class")]
        config = _make_class_config()
        mask = rasterize_polylines(features, 512, 512, config)
        assert np.all(mask == 0)

    def test_single_point_line(self):
        features = [_make_line_feature([[256, 256]], "highway_lane")]
        config = _make_class_config()
        # Should not crash
        mask = rasterize_polylines(features, 512, 512, config)
        assert mask.shape == (512, 512)

    def test_line_at_boundary(self):
        features = [_make_line_feature([[0, 0], [511, 0]], "highway_lane")]
        config = _make_class_config()
        mask = rasterize_polylines(features, 512, 512, config)
        assert mask[0, 256] == 7  # top edge


class TestLoadClassConfig:
    """Test YAML config loading."""

    def test_load_valid_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("classes:\n  road:\n    index: 1\n    buffer_px: 3\n")
            f.flush()
            config = load_class_config(f.name)
            assert "road" in config
            assert config["road"]["index"] == 1
            os.unlink(f.name)
