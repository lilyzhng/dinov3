# Design: DPT & Lightweight Segmentation Heads + Tile Data Pipeline

## Motivation

The existing DINOv3 segmentation evaluation supports Mask2Former (M2F) and linear heads. For satellite/aerial imagery tasks like lane segmentation, we need:

1. **A dense prediction head (DPT)** that leverages multi-scale ViT features with progressive fusion — more expressive than a linear probe, lighter than M2F.
2. **A lightweight MLP head** for fast prototyping and ablation baselines (~2-3M params vs ~20M+ for DPT).
3. **A tile-based data pipeline** for loading satellite/aerial imagery stored as GeoTIFF tiles with rasterized vector labels.
4. **Diversity sampling** to select a representative training subset from large tile pools using DINOv3 features + k-Means.

## Architecture

### DPT Segmentation Head

Adapted from the existing depth DPT head (`dinov3/eval/depth/models/dpt_head.py`).

```
ViT Backbone (frozen)
  │
  ├─ Layer L/4  ──► Reassemble (1x1 proj + 4× upsample) ──► Conv ──► Fusion Block 4
  ├─ Layer L/2  ──► Reassemble (1x1 proj + 2× upsample) ──► Conv ──► Fusion Block 3
  ├─ Layer 3L/4 ──► Reassemble (1x1 proj + identity)     ──► Conv ──► Fusion Block 2
  └─ Layer L    ──► Reassemble (1x1 proj + 2× downsample)──► Conv ──► Fusion Block 1
                                                                           │
                                                              Project (3x3 conv)
                                                                           │
                                                              Seg Head (conv → upsample → conv → ReLU → 1x1)
                                                                           │
                                                              Output: (B, num_classes, H, W)
```

**Key components:**
- `ReassembleBlocks`: Processes CLS tokens (project/add/ignore readout) and rearranges patch tokens into spatial feature maps at 4 different scales.
- `FeatureFusionBlock`: Pre-activation residual conv units + bilinear 2× upsample + 1x1 projection. Progressive bottom-up fusion from deepest to shallowest features.
- `SegmentationHead`: Final conv layers with upsampling, outputs per-pixel class logits.

**Configuration (`config-tiles-dpt.yaml`):**
- `dpt_channels`: 256 (intermediate fusion channels)
- `dpt_post_process_channels`: [128, 256, 512, 1024]
- `dpt_readout_type`: "project" (learns a linear projection of [patch_token, cls_token] concatenation)

### Lightweight MLP Head

A minimal head for fast iteration:

```
ViT Backbone (frozen)
  │
  ├─ Layer L/4  ──► 1x1 Conv (embed_dim → project_dim) ──┐
  ├─ Layer L/2  ──► 1x1 Conv → bilinear upsample        ──┤
  ├─ Layer 3L/4 ──► 1x1 Conv → bilinear upsample        ──┤ Concatenate
  └─ Layer L    ──► 1x1 Conv → bilinear upsample        ──┘
                                                            │
                                          2-layer MLP (1x1 Conv → BN → ReLU → Dropout → 1x1 Conv)
                                                            │
                                          Output: (B, num_classes, H, W)
```

**Design decisions:**
- All spatial features upsampled to the resolution of the shallowest layer (largest spatial size), then concatenated along the channel dimension.
- Uses 1x1 convolutions throughout (no spatial convolutions) — keeps parameter count low.
- Optional CLS token support: when enabled, CLS token is broadcast-concatenated to spatial features before projection.
- SyncBatchNorm for multi-GPU training stability.

### Head Comparison

| Property | DPT | Lightweight | Linear |
|---|---|---|---|
| Parameters | ~20M+ | ~2-3M | ~1M |
| Multi-scale fusion | Progressive residual | Concat + MLP | Concat + linear |
| Spatial convolutions | Yes (3x3) | No (1x1 only) | No |
| CLS token handling | Project readout | Optional concat | No |
| Best for | Final models | Ablations, baselines | Quick probes |

## Data Pipeline

### Tile Dataset

`TileSegmentationDataset` loads pre-tiled satellite/aerial imagery with paired segmentation masks.

**Expected directory structure:**
```
root/
  train/
    images/    # .png, .jpg, .tif, .tiff
    masks/     # matching filenames, single-channel, pixel value = class index
  val/
    images/
    masks/
```

**Features:**
- GeoTIFF support via `rasterio` (falls back to PIL for simple TIFFs)
- Automatic mask filename matching (tries same extension, then .png, then .tif)
- Multi-band normalization to uint8 for non-standard GeoTIFFs

### Rasterization (Vector → Raster)

`rasterize.py` converts GeoJSON polyline/polygon annotations into pixel-level segmentation masks.

**Supported geometry types:** LineString, MultiLineString, Polygon

**Class config (`lane_classes.yaml`):**
```yaml
classes:
  background:    { index: 0, buffer_px: 0 }
  double_solid:  { index: 1, buffer_px: 3 }
  wide_lane:     { index: 2, buffer_px: 5 }
  road_curvature:{ index: 3, buffer_px: 3 }
  road_divider:  { index: 4, buffer_px: 4 }
  intersection:  { index: 5, buffer_px: 0 }  # polygon
  urban_lane:    { index: 6, buffer_px: 3 }
  highway_lane:  { index: 7, buffer_px: 3 }
```

- `buffer_px` controls the line thickness when rasterizing polylines (thicker = easier to learn, but less precise).
- Polygons are filled rather than stroked.
- Geographic → pixel coordinate transform with y-axis flip for geo-referenced tiles.

### Diversity Sampling

Selects a representative subset from a large tile pool:

1. Extract DINOv3 CLS token features for all tiles (frozen backbone, single forward pass).
2. Run MiniBatchKMeans with K = desired subset size.
3. For each cluster, select the tile nearest to the centroid.

**Why k-Means over random sampling?**
- Ensures spatial and visual diversity (avoids oversampling homogeneous regions like empty roads or dense forests).
- Reproducible (seeded).
- Scales to large pools via MiniBatchKMeans.

## Eval Pipeline Changes

### `reduce_zero_label` Configuration

Previously hardcoded to `True` (ADE20K convention: label 0 = unlabeled/background, shifted down by 1 during training).

For lane segmentation, class 0 = background is a meaningful class. The `reduce_zero_label` flag is now:
- Configurable in `EvalConfig` (default: `True` for backward compatibility with ADE20K)
- Threaded through `train.py` → `validate()` → `evaluate_segmentation_model()`
- Set to `false` in tile config YAMLs

## File Summary

```
dinov3/eval/segmentation/
  models/
    __init__.py                      # Updated: build_segmentation_decoder supports "dpt" and "lightweight"
    heads/
      dpt_head.py                    # NEW: DPT segmentation head
      lightweight_head.py            # NEW: Lightweight MLP head
  datasets/
    __init__.py                      # NEW: Package init
    tile_dataset.py                  # NEW: TileSegmentationDataset
    rasterize.py                     # NEW: GeoJSON → mask rasterization
    diversity_sampling.py            # NEW: k-Means diversity sampling
  configs/
    config-tiles-dpt.yaml            # NEW: DPT config for tile segmentation
    config-tiles-lightweight.yaml    # NEW: Lightweight config for tile segmentation
    lane_classes.yaml                # NEW: Lane class definitions
  config.py                          # Updated: DecoderConfig with DPT/lightweight params
  train.py                           # Updated: Supports dpt/lightweight training, reduce_zero_label passthrough
  eval.py                            # Updated: reduce_zero_label parameter
tests/eval/segmentation/
  test_dpt_head.py                   # NEW
  test_lightweight_head.py           # NEW
  test_rasterize.py                  # NEW
  test_diversity_sampling.py         # NEW
  test_data_generator.py             # NEW
  test_training_pipeline.py          # NEW
  test_eval_pipeline.py              # NEW
```
