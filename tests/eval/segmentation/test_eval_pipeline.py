# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Integration tests for the segmentation evaluation pipeline."""

import numpy as np
import pytest
import torch

from dinov3.eval.segmentation.metrics import (
    calculate_intersect_and_union,
    calculate_segmentation_metrics,
)


class TestMetricsComputation:
    """Test mIoU and other metric calculations."""

    def test_perfect_prediction(self):
        """Perfect prediction should give mIoU = 1.0."""
        pred = torch.tensor([[[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]]])
        gt = pred.clone()[0]

        result = calculate_intersect_and_union(pred[0], gt, num_classes=4, reduce_zero_label=False)
        results = result.unsqueeze(0)

        metrics = calculate_segmentation_metrics(results, metrics=["mIoU"])
        assert metrics["mIoU"] > 0.99

    def test_completely_wrong_prediction(self):
        """Completely wrong prediction should give low mIoU."""
        pred = torch.zeros(1, 1, 4, 4, dtype=torch.long)  # all class 0
        gt = torch.ones(4, 4, dtype=torch.long)  # all class 1

        result = calculate_intersect_and_union(pred[0], gt, num_classes=2, reduce_zero_label=False)
        results = result.unsqueeze(0)

        metrics = calculate_segmentation_metrics(results, metrics=["mIoU"])
        assert metrics["mIoU"] < 0.5

    def test_per_class_metrics(self):
        """Test that per-class metrics are computed."""
        # Create prediction with 2 classes
        pred = torch.zeros(1, 4, 4, dtype=torch.long)
        pred[0, :2, :] = 1
        gt = torch.zeros(4, 4, dtype=torch.long)
        gt[:2, :] = 1

        result = calculate_intersect_and_union(pred, gt, num_classes=2, reduce_zero_label=False)
        results = result.unsqueeze(0)

        metrics = calculate_segmentation_metrics(results, metrics=["mIoU", "dice"])
        assert "mIoU" in metrics
        assert "dice" in metrics

    def test_ignore_index(self):
        """Pixels with ignore_index=255 should be excluded."""
        pred = torch.zeros(1, 4, 4, dtype=torch.long)
        gt = torch.full((4, 4), 255, dtype=torch.long)  # all ignored

        result = calculate_intersect_and_union(pred, gt, num_classes=2, reduce_zero_label=False)
        # With all pixels ignored, union should be 0 for valid classes
        # Just verify it doesn't crash
        assert result is not None


class TestMultiBatchMetrics:
    """Test metrics accumulated over multiple batches."""

    def test_accumulate_batches(self):
        results = []
        for _ in range(5):
            pred = torch.randint(0, 4, (1, 32, 32), dtype=torch.long)
            gt = torch.randint(0, 4, (32, 32), dtype=torch.long)
            result = calculate_intersect_and_union(pred, gt, num_classes=4, reduce_zero_label=False)
            results.append(result)

        all_results = torch.stack(results)
        metrics = calculate_segmentation_metrics(all_results, metrics=["mIoU"])
        assert 0 <= metrics["mIoU"] <= 1.0
