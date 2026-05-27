from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_index: int | None = None) -> tuple[torch.Tensor, int, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        score = logits[:, class_index].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach(), class_index, logits.detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()
