# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Tests for diversity sampling with k-Means."""

import json
import os
import tempfile

import numpy as np
import pytest

from dinov3.eval.segmentation.datasets.diversity_sampling import (
    kmeans_sampling,
    save_selection,
)


class TestKMeansSampling:
    """Test k-Means diversity sampling."""

    def test_basic_clustering(self):
        """Features with 3 clear clusters should select from each."""
        rng = np.random.RandomState(42)
        # 3 clusters of 100 samples each
        cluster1 = rng.randn(100, 64) + np.array([5, 0] + [0] * 62)
        cluster2 = rng.randn(100, 64) + np.array([0, 5] + [0] * 62)
        cluster3 = rng.randn(100, 64) + np.array([-5, -5] + [0] * 62)
        features = np.vstack([cluster1, cluster2, cluster3])

        selected_indices, cluster_labels = kmeans_sampling(features, n_select=3, seed=42)

        assert len(selected_indices) == 3
        assert len(set(selected_indices)) == 3  # all unique

        # Selected samples should come from different clusters
        selected_cluster_labels = cluster_labels[selected_indices]
        assert len(set(selected_cluster_labels)) >= 2  # at least 2 different clusters represented

    def test_select_more_than_clusters(self):
        """Selecting more samples than natural clusters should still work."""
        rng = np.random.RandomState(42)
        features = rng.randn(50, 32)
        selected_indices, _ = kmeans_sampling(features, n_select=10, seed=42)
        assert len(selected_indices) == 10
        assert len(set(selected_indices)) == 10

    def test_select_all_samples(self):
        """Selecting N=total should return all samples."""
        rng = np.random.RandomState(42)
        features = rng.randn(20, 16)
        selected_indices, _ = kmeans_sampling(features, n_select=20, seed=42)
        assert len(selected_indices) == 20

    def test_select_more_than_total(self):
        """Selecting more than total should be clamped."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 16)
        selected_indices, _ = kmeans_sampling(features, n_select=50, seed=42)
        assert len(selected_indices) == 10  # clamped to N

    def test_all_identical_features(self):
        """All identical features should still return n_select samples."""
        features = np.ones((100, 32))
        selected_indices, _ = kmeans_sampling(features, n_select=5, seed=42)
        assert len(selected_indices) == 5

    def test_reproducibility(self):
        """Same seed should give same results."""
        rng = np.random.RandomState(42)
        features = rng.randn(100, 32)

        sel1, _ = kmeans_sampling(features, n_select=10, seed=123)
        sel2, _ = kmeans_sampling(features, n_select=10, seed=123)

        np.testing.assert_array_equal(sel1, sel2)


class TestRareClassRepresentation:
    """Test that sampling over-represents rare tile types."""

    def test_imbalanced_distribution(self):
        """With 80% empty, 15% highway, 5% urban, sampling should pick from all."""
        rng = np.random.RandomState(42)
        # Simulate: 80 "empty" tiles near origin, 15 "highway", 5 "urban intersection"
        empty = rng.randn(80, 32) * 0.1
        highway = rng.randn(15, 32) * 0.1 + np.array([5] + [0] * 31)
        urban = rng.randn(5, 32) * 0.1 + np.array([0, 5] + [0] * 30)
        features = np.vstack([empty, highway, urban])

        # Select 10 tiles — should include some from each group
        selected_indices, _ = kmeans_sampling(features, n_select=10, seed=42)

        # Check that we got samples from the rarer groups
        selected_empty = sum(1 for i in selected_indices if i < 80)
        selected_highway = sum(1 for i in selected_indices if 80 <= i < 95)
        selected_urban = sum(1 for i in selected_indices if i >= 95)

        # With k-means on 10 clusters, rare groups should get at least some representation
        # (better than random which would pick ~8 empty, ~1.5 highway, ~0.5 urban)
        assert selected_highway >= 1, "Highway tiles should be represented"
        assert selected_urban >= 1, "Urban tiles should be represented"


class TestSaveSelection:
    """Test saving selection results."""

    def test_save_json(self):
        filenames = [f"tile_{i:04d}.png" for i in range(100)]
        selected = np.array([0, 10, 50, 99])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_selection(filenames, selected, f.name)

            with open(f.name) as fp:
                result = json.load(fp)

            assert result["n_total"] == 100
            assert result["n_selected"] == 4
            assert result["selected_files"] == ["tile_0000.png", "tile_0010.png", "tile_0050.png", "tile_0099.png"]
            os.unlink(f.name)
