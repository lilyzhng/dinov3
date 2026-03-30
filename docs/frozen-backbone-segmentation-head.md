# Segmentation Head: Frozen Backbone + DPT Head (No LoRA Required)

## Summary

For our lane line segmentation task, we use a **frozen DINOv3 backbone** as a feature extractor and train a **randomly initialized DPT segmentation head** from scratch. This is the same proven approach used by the existing DINOv3 depth estimation pipeline. **No pre-trained DPT weights or LoRA fine-tuning are needed.**

## Architecture

```
DINOv3 Backbone (frozen, pre-trained) --> 4 intermediate features --> DPT Segmentation Head (trainable, random init) --> per-pixel class logits
```

## Key Evidence: Frozen Backbone Works

The existing DINOv3 depth task already proves this approach. The backbone is **explicitly frozen** in the depth encoder wrapper:

- [`dinov3/eval/depth/models/encoder.py` (line 103)](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/eval/depth/models/encoder.py#L103) -- `self.backbone.requires_grad_(False)`

The full encoder wrapper that freezes the backbone and extracts intermediate features:

- [`dinov3/eval/depth/models/encoder.py` -- `DinoVisionTransformerWrapper`](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/eval/depth/models/encoder.py#L61-L117)

Pre-trained depth head weights (trained on top of this frozen backbone) are published and ship as official models:

- [`dinov3/hub/depthers.py` -- factory functions + weight loading](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/hub/depthers.py)

## Existing Code References

### DPT Head Implementations

| Task | File | Description |
|------|------|-------------|
| Depth | [`dinov3/eval/depth/models/dpt_head.py`](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/eval/depth/models/dpt_head.py) | DPT head for depth estimation (~20-30M params) |
| Segmentation | [`dinov3/eval/segmentation/models/heads/dpt_head.py`](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/eval/segmentation/models/heads/dpt_head.py) | DPT head for segmentation (same architecture, different output) |

### Backbone + Head Wiring

| Component | File | Description |
|-----------|------|-------------|
| Backbone freeze | [`encoder.py#L103`](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/eval/depth/models/encoder.py#L103) | `self.backbone.requires_grad_(False)` |
| Intermediate feature extraction | [`encoder.py#L105-L117`](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/eval/depth/models/encoder.py#L105-L117) | Extracts 4 intermediate layers from frozen backbone |
| Decoder builder | [`dinov3/eval/depth/models/__init__.py`](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/eval/depth/models/__init__.py) | `build_depther()`, `DecoderConfig`, `FeaturesToDepth` |
| Pre-trained depth weights | [`dinov3/hub/depthers.py`](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/hub/depthers.py) | Published weights trained with this frozen-backbone approach |

### Segmentation Training Pipeline

| Component | File | Description |
|-----------|------|-------------|
| Segmentation decoder builder | [`dinov3/eval/segmentation/models/__init__.py`](https://github.com/lilyzhng/dinov3/blob/347cb0233ad48352365f910266d270097ab83f9b/dinov3/eval/segmentation/models/__init__.py) | `build_segmentation_decoder()` |
| Available heads | DPT, Lightweight (~2-3M), Linear (~1M) | Multiple options depending on capacity needs |

## Why LoRA Does Not Apply Here

LoRA (Low-Rank Adaptation) is designed for efficiently fine-tuning **large pre-trained models** by adding low-rank weight updates instead of training all parameters. It does not make sense for our segmentation head because:

1. **The head is initialized from scratch** -- there are no pre-trained weights to adapt with LoRA.
2. **The head is relatively small** (~20-30M params for DPT) -- we can train all parameters directly.
3. **The head is already the only trainable component** -- the backbone is frozen, so there is no large model to efficiently fine-tune.
4. **LoRA would constrain capacity** -- low-rank updates would limit what the head can learn, for zero benefit.

LoRA would only make sense if we were fine-tuning the **backbone itself** (billions of parameters, expensive full fine-tuning). But we are not -- and the existing depth pipeline proves the frozen backbone approach works.

## Why Frozen DINOv3 Features Are Sufficient

The historical concern that "frozen features don't work" comes from the era of supervised CNN backbones (e.g., ResNet trained on ImageNet classification). Those features were task-specific and transferred poorly.

DINOv3 is fundamentally different:

- Trained with **self-supervised learning** (DINO objective) on massive diverse data
- Produces **general-purpose features** not tied to any specific downstream task
- The same frozen backbone already powers depth, segmentation (ADE20K), detection, and other tasks successfully

## Resolution Requirements

For lane line segmentation specifically (dashed vs solid, single vs double), we need **12-13 cm/pixel minimum resolution**. At 15 cm/pixel, sub-pixel distinctions become nearly impossible. This is independent of the backbone/head architecture question.
