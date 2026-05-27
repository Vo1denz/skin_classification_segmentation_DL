from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skin_cancer_dl.models import build_classifier, build_unet


def main() -> None:
    unet = build_unet(base_channels=8)
    seg_out = unet(torch.randn(2, 3, 128, 128))
    assert seg_out.shape == (2, 1, 128, 128)

    classifier = build_classifier("mobilenet_v2", num_classes=3, pretrained=False).model
    cls_out = classifier(torch.randn(2, 3, 224, 224))
    assert cls_out.shape == (2, 3)

    print("smoke test passed")


if __name__ == "__main__":
    main()
