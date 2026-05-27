from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .models import build_classifier, build_unet
from .utils import (
    apply_mask,
    image_to_tensor,
    load_checkpoint,
    overlay_heatmap,
    tensor_mask_to_numpy,
    top_k_predictions,
)
from .xai import GradCAM


class SkinCancerPipeline:
    def __init__(
        self,
        segmentation_model: torch.nn.Module | None,
        classifier_model: torch.nn.Module,
        classifier_target_layer: torch.nn.Module,
        class_names: list[str],
        segmentation_size: int,
        classification_size: int,
        device: torch.device,
    ) -> None:
        self.segmentation_model = segmentation_model
        self.classifier_model = classifier_model
        self.classifier_target_layer = classifier_target_layer
        self.class_names = class_names
        self.segmentation_size = segmentation_size
        self.classification_size = classification_size
        self.device = device

        if self.segmentation_model is not None:
            self.segmentation_model.eval()
        self.classifier_model.eval()

    @classmethod
    def from_checkpoints(
        cls,
        classifier_checkpoint: str | Path,
        segmentation_checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> "SkinCancerPipeline":
        device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        segmentation_model = None
        segmentation_size = 256
        if segmentation_checkpoint:
            seg_payload = load_checkpoint(segmentation_checkpoint, device)
            segmentation_size = int(seg_payload.get("image_size", 256))
            base_channels = int(seg_payload.get("base_channels", 32))
            segmentation_model = build_unet(base_channels=base_channels).to(device)
            segmentation_model.load_state_dict(seg_payload["model_state"])

        cls_payload = load_checkpoint(classifier_checkpoint, device)
        model_name = cls_payload.get("model_name", "efficientnet_b0")
        class_names = list(cls_payload["class_names"])
        classification_size = int(cls_payload.get("image_size", 224))
        spec = build_classifier(model_name, num_classes=len(class_names), pretrained=False)
        classifier_model = spec.model.to(device)
        classifier_model.load_state_dict(cls_payload["model_state"])

        return cls(
            segmentation_model=segmentation_model,
            classifier_model=classifier_model,
            classifier_target_layer=spec.target_layer,
            class_names=class_names,
            segmentation_size=segmentation_size,
            classification_size=classification_size,
            device=device,
        )

    @torch.no_grad()
    def segment(self, image_rgb: np.ndarray) -> np.ndarray | None:
        if self.segmentation_model is None:
            return None
        height, width = image_rgb.shape[:2]
        tensor = image_to_tensor(image_rgb, size=self.segmentation_size).unsqueeze(0).to(self.device)
        logits = self.segmentation_model(tensor)
        probs = torch.sigmoid(logits)
        return tensor_mask_to_numpy(probs, height=height, width=width)

    def classify(self, image_rgb: np.ndarray, top_k: int = 3, explain: bool = True) -> dict[str, Any]:
        tensor = image_to_tensor(image_rgb, size=self.classification_size).unsqueeze(0).to(self.device)

        if explain:
            gradcam = GradCAM(self.classifier_model, self.classifier_target_layer)
            heatmap_tensor, predicted_index, logits = gradcam(tensor)
            gradcam.close()
            heatmap = heatmap_tensor.detach().cpu().numpy()
        else:
            with torch.no_grad():
                logits = self.classifier_model(tensor)
            predicted_index = int(logits.argmax(dim=1).item())
            heatmap = None

        probs = F.softmax(logits[0], dim=0)
        result: dict[str, Any] = {
            "predicted_class": self.class_names[predicted_index],
            "confidence": float(probs[predicted_index].detach().cpu()),
            "top_predictions": top_k_predictions(probs, self.class_names, k=top_k),
            "heatmap": heatmap,
        }
        return result

    def predict(self, image_rgb: np.ndarray, top_k: int = 3, explain: bool = True) -> dict[str, Any]:
        mask = self.segment(image_rgb)
        model_input = image_rgb

        if mask is not None:
            binary_mask = (mask > 0.5).astype(np.float32)
            model_input = apply_mask(image_rgb, binary_mask)

        classification = self.classify(model_input, top_k=top_k, explain=explain)
        heatmap = classification.pop("heatmap")

        gradcam_overlay = None
        if heatmap is not None:
            heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            gradcam_overlay = overlay_heatmap(model_input, heatmap)

        return {
            "mask": mask,
            "segmented_image": model_input,
            "gradcam_overlay": gradcam_overlay,
            "prediction": classification,
        }
