# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Unit tests for the DPT segmentation head."""

import pytest
import torch

from dinov3.eval.segmentation.models.heads.dpt_head import DPTHead


def _make_dummy_inputs(batch_size=2, embed_dim=1024, h=32, w=32, n_levels=4):
    """Create dummy backbone outputs: list of (features, cls_token) tuples."""
    return [
        (torch.randn(batch_size, embed_dim, h, w), torch.randn(batch_size, embed_dim))
        for _ in range(n_levels)
    ]


class TestDPTHeadOutputShape:
    """Test that DPTHead produces correct output shapes."""

    @pytest.mark.parametrize("num_classes", [2, 8, 20])
    def test_output_shape(self, num_classes):
        head = DPTHead(
            in_channels=(1024, 1024, 1024, 1024),
            n_output_channels=num_classes,
            readout_type="project",
        )
        inputs = _make_dummy_inputs()
        output = head(inputs)
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == num_classes
        # Spatial dims should be larger than input due to upsampling
        assert output.shape[2] > 0
        assert output.shape[3] > 0

    def test_output_shape_ignore_readout(self):
        head = DPTHead(
            in_channels=(1024, 1024, 1024, 1024),
            n_output_channels=8,
            readout_type="ignore",
        )
        inputs = _make_dummy_inputs()
        output = head(inputs)
        assert output.shape[1] == 8

    def test_output_shape_add_readout(self):
        head = DPTHead(
            in_channels=(1024, 1024, 1024, 1024),
            n_output_channels=8,
            readout_type="add",
        )
        inputs = _make_dummy_inputs()
        output = head(inputs)
        assert output.shape[1] == 8


class TestDPTHeadParameters:
    """Test parameter count and structure."""

    def test_param_count_reasonable(self):
        head = DPTHead(
            in_channels=(1024, 1024, 1024, 1024),
            n_output_channels=8,
        )
        n_params = sum(p.numel() for p in head.parameters())
        # DPT head should have significant params (> 1M, < 100M)
        assert n_params > 1_000_000
        assert n_params < 100_000_000

    def test_all_params_have_grad(self):
        head = DPTHead(
            in_channels=(1024, 1024, 1024, 1024),
            n_output_channels=8,
        )
        for name, param in head.named_parameters():
            assert param.requires_grad, f"Parameter {name} should require grad"


class TestDPTHeadGradientFlow:
    """Test that gradients flow correctly."""

    def test_backward_pass(self):
        head = DPTHead(
            in_channels=(384, 384, 384, 384),
            channels=128,
            post_process_channels=[64, 128, 256, 512],
            n_output_channels=8,
        )
        inputs = _make_dummy_inputs(batch_size=1, embed_dim=384, h=16, w=16)
        output = head(inputs)
        loss = output.sum()
        loss.backward()

        # Check that at least some parameters got gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
        assert has_grad, "No gradients found in any parameter"


class TestDPTHeadPredict:
    """Test the predict method with rescaling."""

    def test_predict_rescale(self):
        head = DPTHead(
            in_channels=(384, 384, 384, 384),
            channels=128,
            post_process_channels=[64, 128, 256, 512],
            n_output_channels=8,
        )
        inputs = _make_dummy_inputs(batch_size=1, embed_dim=384, h=16, w=16)
        output = head.predict(inputs, rescale_to=(256, 256))
        assert output.shape == (1, 8, 256, 256)

    def test_predict_different_rescale_sizes(self):
        head = DPTHead(
            in_channels=(384, 384, 384, 384),
            channels=128,
            post_process_channels=[64, 128, 256, 512],
            n_output_channels=8,
        )
        inputs = _make_dummy_inputs(batch_size=1, embed_dim=384, h=16, w=16)
        for target_h, target_w in [(128, 128), (512, 512), (224, 224)]:
            output = head.predict(inputs, rescale_to=(target_h, target_w))
            assert output.shape == (1, 8, target_h, target_w)


class TestDPTHeadSmallModel:
    """Test with smaller dimensions for faster testing."""

    def test_small_model_forward(self):
        head = DPTHead(
            in_channels=(64, 64, 64, 64),
            channels=32,
            post_process_channels=[16, 32, 64, 128],
            n_output_channels=2,
            n_hidden_channels=8,
        )
        inputs = _make_dummy_inputs(batch_size=1, embed_dim=64, h=8, w=8)
        output = head(inputs)
        assert output.shape[0] == 1
        assert output.shape[1] == 2
