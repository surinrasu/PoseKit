import argparse
import os
import sys
from typing import Any, Dict

import torch
import torch.nn as nn

from lib.models import PoseKitModel, Backbone, Header

try:
    import coremltools as ct
except Exception as exc:
    ct = None
    _coreml_import_error = exc

class BackboneNoNorm(Backbone):
    def forward(self, x):
        f1 = self.features1(x)
        f2 = self.features2(f1)
        f3 = self.features3(f2)

        f4 = self.features4(f3)
        f3 = self.conv3(f3)
        f4 = f4 + f3

        f4 = self.upsample2(f4)
        f2 = self.conv2(f2)
        f4 = f4 + f2

        f4 = self.upsample1(f4)
        f1 = self.conv1(f1)
        f4 = f4 + f1

        f4 = self.conv4(f4)
        return f4

class PoseKitModelNoNorm(nn.Module):
    def __init__(self, num_classes: int = 17, mode: str = "train"):
        super().__init__()
        self.backbone = BackboneNoNorm()
        self.header = Header(num_classes, mode)

    def forward(self, x):
        x = self.backbone(x)
        out = self.header(x)
        return tuple(out)

class PoseKitModelWrapped(nn.Module):
    def __init__(self, num_classes: int = 17, mode: str = "train"):
        super().__init__()
        self.model = PoseKitModel(num_classes=num_classes, mode=mode)

    def forward(self, x):
        out = self.model(x)
        return tuple(out)

def _load_state_dict(weights_path: str) -> Dict[str, Any]:
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise ValueError(f"Unrecognized checkpoint format in {weights_path}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert PoseKitModel to Core ML")
    p.add_argument("--weights", default="../output/last.pth", help="Path to .pth checkpoint")
    p.add_argument("--output", default="../output/posekit.mlpackage", help="Output .mlpackage path")
    p.add_argument("--input-size", type=int, default=192, help="Square input size")
    p.add_argument("--num-classes", type=int, default=17, help="Number of keypoints")
    p.add_argument("--keep-norm-in-graph", action="store_true", help="Keep x/127.5-1 in the graph")
    p.add_argument("--use-coreml-preprocess", dest="use_coreml_preprocess", action="store_true", default=True,
                   help="Use Core ML image preprocessing (scale/bias). Only valid if --keep-norm-in-graph is NOT set.")
    p.add_argument("--no-coreml-preprocess", dest="use_coreml_preprocess", action="store_false",
                   help="Disable Core ML image preprocessing (expects inputs already normalized to [-1, 1]).")
    p.add_argument("--color-layout", choices=["RGB", "BGR"], default="RGB")
    p.add_argument("--ios-target", default="18", choices=["15", "16", "17", "18"],
                   help="Minimum iOS deployment target for ML Program")
    return p.parse_args()

def main() -> int:
    args = parse_args()

    if args.keep_norm_in_graph and args.use_coreml_preprocess:
        print("--use-coreml-preprocess cannot be used with --keep-norm-in-graph", file=sys.stderr)
        return 2
    if (not args.keep_norm_in_graph) and (not args.use_coreml_preprocess):
        print(
            "[WARN] No in-graph normalization and no Core ML preprocessing. "
            "Inputs must already be normalized to [-1, 1].",
            file=sys.stderr,
        )

    if not os.path.exists(args.weights):
        print(f"Weights not found: {args.weights}", file=sys.stderr)
        return 2

    if args.keep_norm_in_graph:
        model = PoseKitModelWrapped(num_classes=args.num_classes, mode="train")
    else:
        model = PoseKitModelNoNorm(num_classes=args.num_classes, mode="train")

    state_dict = _load_state_dict(args.weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    example = torch.randn(1, 3, args.input_size, args.input_size)
    traced = torch.jit.trace(model, example)
    traced = torch.jit.freeze(traced)

    if args.use_coreml_preprocess:
        color_layout = ct.colorlayout.RGB if args.color_layout == "RGB" else ct.colorlayout.BGR
        inputs = [ct.ImageType(
            name="input",
            shape=example.shape,
            scale=1.0 / 127.5,
            bias=[-1.0, -1.0, -1.0],
            color_layout=color_layout,
        )]
    else:
        inputs = [ct.TensorType(name="input", shape=example.shape)]

    outputs = [
        ct.TensorType(name="heatmaps"),
        ct.TensorType(name="centers"),
        ct.TensorType(name="regs"),
        ct.TensorType(name="offsets"),
    ]

    ios_target = {
        "15": ct.target.iOS15,
        "16": ct.target.iOS16,
        "17": ct.target.iOS17,
        "18": ct.target.iOS18,
    }[args.ios_target]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ios_target,
        compute_units=ct.ComputeUnit.ALL,
    )

    out_path = args.output
    if not out_path.endswith(".mlpackage"):
        out_path = out_path + ".mlpackage"

    mlmodel.save(out_path)
    print(f"Saved Core ML model to: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
