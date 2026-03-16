# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# Lightweight MLP segmentation head inspired by SegDINO.
# Takes 4 intermediate ViT features, projects and fuses them with a simple MLP.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightHead(nn.Module):
    """Lightweight MLP-based segmentation head.

    Projects each of the 4 intermediate features to a common dimension,
    upsamples to the same spatial resolution, concatenates, and applies
    a 2-layer MLP to produce per-pixel class logits.

    ~2-3M trainable parameters (vs ~20M+ for DPT).
    """

    def __init__(
        self,
        in_channels,
        n_output_channels,
        project_dim=256,
        hidden_dim=512,
        use_batchnorm=True,
        use_cls_token=False,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_output_channels = n_output_channels
        self.use_cls_token = use_cls_token

        # 1x1 conv to project each feature level to common dim
        self.projections = nn.ModuleList([
            nn.Conv2d(ch * (2 if use_cls_token else 1), project_dim, kernel_size=1)
            for ch in in_channels
        ])

        concat_dim = project_dim * len(in_channels)

        # 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(concat_dim, hidden_dim, kernel_size=1),
            nn.SyncBatchNorm(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, n_output_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for proj in self.projections:
            nn.init.kaiming_normal_(proj.weight)
            nn.init.constant_(proj.bias, 0)
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def _process_inputs(self, inputs):
        """Process backbone features, handling optional cls tokens."""
        processed = []
        for i, x in enumerate(inputs):
            if self.use_cls_token:
                assert len(x) == 2, "Missing class tokens"
                feat, cls_token = x[0], x[1]
                cls_token = cls_token[:, :, None, None].expand_as(feat)
                feat = torch.cat((feat, cls_token), 1)
            else:
                feat = x
            processed.append(feat)
        return processed

    def forward(self, inputs):
        features = self._process_inputs(inputs)
        target_size = features[0].shape[2:]

        projected = []
        for i, feat in enumerate(features):
            feat = self.projections[i](feat)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feat)

        x = torch.cat(projected, dim=1)
        return self.mlp(x)

    def predict(self, inputs, rescale_to=(512, 512)):
        out = self.forward(inputs)
        out = F.interpolate(input=out, size=rescale_to, mode="bilinear", align_corners=False)
        return out
