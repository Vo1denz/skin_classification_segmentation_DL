from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def read_image_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


def read_image_rgb_from_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    mask_uint8 = np.clip(mask * 255 if mask.max() <= 1.0 else mask, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), mask_uint8)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def image_to_tensor(image_rgb: np.ndarray, size: int | None = None) -> torch.Tensor:
    if size is not None:
        image_rgb = resize_image(image_rgb, size)
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()


def mask_to_tensor(mask: np.ndarray, size: int | None = None) -> torch.Tensor:
    if size is not None:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (mask.astype(np.float32) > 127.0).astype(np.float32)
    return torch.from_numpy(mask[None, ...]).float()


def tensor_mask_to_numpy(mask: torch.Tensor, height: int, width: int) -> np.ndarray:
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3:
        mask = mask[0]
    mask_np = mask.detach().cpu().numpy()
    mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask_np, 0.0, 1.0)


def apply_mask(image_rgb: np.ndarray, mask: np.ndarray, background: int = 20) -> np.ndarray:
    mask_3 = np.repeat(mask[..., None], 3, axis=2)
    base = np.full_like(image_rgb, background)
    segmented = image_rgb.astype(np.float32) * mask_3 + base.astype(np.float32) * (1.0 - mask_3)
    return np.clip(segmented, 0, 255).astype(np.uint8)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    heat = np.zeros_like(image_rgb)
    heat[..., 0] = 255
    mask_3 = np.repeat(mask[..., None], 3, axis=2)
    overlay = image_rgb.astype(np.float32) * (1.0 - alpha * mask_3)
    overlay += heat.astype(np.float32) * (alpha * mask_3)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    out = image_rgb.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean()


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device)


def top_k_predictions(probs: torch.Tensor, class_names: Iterable[str], k: int = 3) -> list[dict[str, Any]]:
    class_names = list(class_names)
    k = min(k, probs.numel())
    values, indices = torch.topk(probs.detach().cpu(), k=k)
    return [
        {"class": class_names[int(idx)], "probability": float(value)}
        for value, idx in zip(values, indices)
    ]


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count
