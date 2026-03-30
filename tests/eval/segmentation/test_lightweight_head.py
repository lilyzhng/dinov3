# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Unit tests for the lightweight MLP segmentation head."""

import pytest
import torch

from dinov3.eval.segmentation.models.heads.lightweight_head import LightweightHead


def _make_dummy_inputs(batch_size=2, embed_dim=1024, h=32, w=32, n_levels=4):
    """Create dummy backbone outputs: list of feature tensors (no cls token)."""
    return [torch.randn(batch_size, embed_dim, h, w) for _ in range(n_levels)]


class TestLightweightHeadOutputShape:
    """Test that LightweightHead produces correct output shapes."""

    @pytest.mark.parametrize("num_classes", [2, 8, 20])
    def test_output_shape(self, num_classes):
        head = LightweightHead(
            in_channels=[1024, 1024, 1024, 1024],
            n_output_channels=num_classes,
        )
        inputs = _make_dummy_inputs()
        output = head(inputs)
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == num_classes
        assert output.shape[2] == 32  # same as input spatial
        assert output.shape[3] == 32

    def test_output_shape_different_embed_dims(self):
        head = LightweightHead(
            in_channels=[256, 512, 768, 1024],
            n_output_channels=8,
        )
        inputs = [
            torch.randn(1, 256, 32, 32),
            torch.randn(1, 512, 32, 32),
            torch.randn(1, 768, 32, 32),
            torch.randn(1, 1024, 32, 32),
        ]
        output = head(inputs)
        assert output.shape == (1, 8, 32, 32)

    def test_single_level_input(self):
        head = LightweightHead(
            in_channels=[1024],
            n_output_channels=8,
        )
        inputs = [torch.randn(1, 1024, 32, 32)]
        output = head(inputs)
        assert output.shape == (1, 8, 32, 32)


class TestLightweightHeadParameters:
    """Test parameter counts — should be much smaller than DPT."""

    def test_param_count_small(self):
        head = LightweightHead(
            in_channels=[1024, 1024, 1024, 1024],
            n_output_channels=8,
            project_dim=256,
            hidden_dim=512,
        )
        n_params = sum(p.numel() for p in head.parameters())
        # Should be in 1-5M range
        assert n_params < 10_000_000, f"Too many params: {n_params}"

    def test_fewer_params_than_dpt(self):
        from dinov3.eval.segmentation.models.heads.dpt_head import DPTHead

        lightweight = LightweightHead(
            in_channels=[1024, 1024, 1024, 1024],
            n_output_channels=8,
        )
        dpt = DPTHead(
            in_channels=(1024, 1024, 1024, 1024),
            n_output_channels=8,
        )
        lw_params = sum(p.numel() for p in lightweight.parameters())
        dpt_params = sum(p.numel() for p in dpt.parameters())
        assert lw_params < dpt_params, f"Lightweight ({lw_params}) should have fewer params than DPT ({dpt_params})"


class TestLightweightHeadGradientFlow:
    """Test gradient flow."""

    def test_backward_pass(self):
        head = LightweightHead(
            in_channels=[384, 384, 384, 384],
            n_output_channels=8,
            project_dim=64,
            hidden_dim=128,
        )
        inputs = _make_dummy_inputs(batch_size=1, embed_dim=384, h=16, w=16)
        output = head(inputs)
        loss = output.sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
        assert has_grad, "No gradients found"


class TestLightweightHeadPredict:
    """Test predict with rescaling."""

    def test_predict_rescale(self):
        head = LightweightHead(
            in_channels=[384, 384, 384, 384],
            n_output_channels=8,
            project_dim=64,
            hidden_dim=128,
        )
        inputs = _make_dummy_inputs(batch_size=1, embed_dim=384, h=16, w=16)
        output = head.predict(inputs, rescale_to=(256, 256))
        assert output.shape == (1, 8, 256, 256)
