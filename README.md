# PoseKit

![](banner.png)

PoseKit is a lightweight RGB-D 3D human pose estimation system designed for mobile devices. It combines a compact 2D keypoint detector with depth-based 3D reconstruction, and ships with an iOS app that performs real-time inference, smoothing, playback, and export. The design and system rationale are documented in the technical report at `PoseKitPaper/main.tex`.

## Highlights

- **Mobile-first 2D detector**: MobileNetV2-style backbone with depthwise separable convolutions and inverted residual blocks, producing heatmaps, center heatmaps, and offset regressions.
- **Graph-outside decoding**: Peak finding and dynamic indexing are performed on CPU after inference for better hardware optimization in Core ML / ANE.
- **RGB-D 3D recovery**: 2D joints are back-projected into 3D using ARKit depth maps and camera intrinsics.
- **On-device pipeline**: Smoothing, pelvis-centered canonicalization, and ARKit capture are integrated in the iOS app.
- **Export & replay**: Recorded takes can be exported as JSON, binary, or ZIP.

## Overview

1. **2D keypoint detection**
   - Input images are letterboxed to a fixed size (default 192x192).
   - The model predicts:
     - `heatmaps`: per-joint heatmaps.
     - `centers`: person center heatmap.
     - `regs`: center-to-joint coarse offsets.
     - `offsets`: sub-pixel refinements.

2. **Graph-outside decoding**
   - The app finds the strongest center point, then refines each joint by combining heatmaps, center offsets, and local offsets.
   - This matches the Python decoder in `PoseKitModel/pkmodel/lib/task.py` and the Swift decoder in `PoseKitApp/Sources/PoseKitApp/PoseModel/PoseKitModel.swift`.

3. **RGB-D 3D reconstruction**
   - If depth is available, each 2D joint is sampled in the depth map (with confidence filtering) and back-projected using camera intrinsics.
   - When depth is missing or low-confidence, 3D joints are left invalid and the system can still return 2D pose.

4. **Post-processing**
   - Exponential moving average smoothing is applied to 3D joints.
   - A canonical pose is computed in pelvis-centered coordinates for playback.

## Structure

- `PoseKitPaper/`
  - Technical report
- `PoseKitModel/`
  - PyTorch training/evaluation, data prep, Core ML export
- `PoseKitApp/`
  - iOS app, Core ML inference, ARKit capture, playback/export
- `dataset/`
  - Prepared samples and labels: `imgs/`, `train.json`, `val.json`, `label.json`
- `coco2017/`
  - COCO layout (expected for label preparation)
- `aist/`, `yoga82/`
  - URL lists for additional datasets

## Model Details

- **Input**: 192x192 RGB (letterboxed), normalized to [-1, 1].
- **Output grid**: 48x48 (input size / 4).
- **Heads**:
  - Heatmaps: `K x 48 x 48`
  - Center heatmap: `1 x 48 x 48`
  - Coarse offsets: `2K x 48 x 48`
  - Sub-pixel offsets: `2K x 48 x 48`
- **Keypoints**: COCO-17 plus derived pelvis/neck centers.

## Requirements

**Model**:
- Python 3.12 - 3.14
- PyTorch, torchvision, OpenCV, NumPy, Albumentations, coremltools

**App**:
- iOS 17 target
- LiDAR-capable devices

## Build

The training code uses Poetry and a `pkmodel` CLI, you can use the pre-defined mise tasks as well:

```bash
# Setup
mise trust
mise i

mise r prepare-coco # Prepare COCO dataset
mise r train # Train
mise r evaluate # Evaluate
```

Default paths:

- Images: `../../coco2017/train2017`, `../../coco2017/val2017`
- Labels: `../../coco2017/annotations/person_keypoints_*.json`
- Output: `../../dataset/imgs`, `../../dataset/train.json`, `../../dataset/val.json`

You may change them and other config entries in `config.toml` or the `env` section of `mise.toml`(or of course by setting env variables).

To run the iOS app, you need to convert a PyTorch checkpoint to an CoreML model and compile it first:

```bash
mise r generate-coreml
mise r compile-coreml
```

The Core ML preprocessing applies `scale=1/127.5` and `bias=-1` for RGB by default.

Then connect your iPhone, trust your mac, turn on developer mode, maybe a reboot, trust your developer account, and run:

```bash
mise r run
# or if you have mutiple destinations
mise r run "xxx's iPhone"
```

## Technical Notes

- Decoupling 2D keypoint detection from 3D recovery to keep the network light.
- Using CPU-side decoding to avoid dynamic ops in the inference graph.
- Leveraging depth sensors and camera intrinsics for 3D reconstruction.
- Deployment considerations for Core ML / ML Program and mobile hardware.
- Discussion of compression options (FP16, quantization, pruning, distillation) and cross-platform fallbacks when depth is unavailable.

The report at `PoseKitPaper/main.tex` describes more about the system design and engineering choices, get a more human-friendly version by:

```bash
mise r generate-paper
```

## Limitations

- 3D reconstruction depends on depth availability and quality.
- Without this, the system falls back to 2D keypoints only.
- Cross-platform depth alignment and calibration also remain difficult to guarantee.
