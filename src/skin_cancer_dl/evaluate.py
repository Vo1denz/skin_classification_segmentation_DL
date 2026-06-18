"""Evaluate the trained classifier on a dataset and produce confusion matrix + metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from .datasets import classification_transforms
from .models import build_classifier
from .utils import load_checkpoint


def evaluate_classifier(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run the classifier on every image in *data_dir* and return metrics.

    Parameters
    ----------
    checkpoint_path:
        Path to the classifier ``.pt`` checkpoint.
    data_dir:
        An ``ImageFolder``-style directory (sub-folders = class names).
    batch_size:
        Batch size for evaluation.
    num_workers:
        DataLoader workers.
    device:
        Torch device.  Defaults to CUDA when available.

    Returns
    -------
    dict with keys:
        class_names, confusion_matrix, metrics, total_samples
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # ── load checkpoint ──────────────────────────────────────────────
    payload = load_checkpoint(checkpoint_path, device)
    model_name: str = payload.get("model_name", "efficientnet_b0")
    class_names: list[str] = list(payload["class_names"])
    image_size: int = int(payload.get("image_size", 224))
    num_classes = len(class_names)

    spec = build_classifier(model_name, num_classes=num_classes, pretrained=False)
    model = spec.model.to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    # ── build dataset / loader ───────────────────────────────────────
    transform = classification_transforms(image_size, train=False)
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)

    # Ensure folder class order matches checkpoint class order
    folder_classes = dataset.classes
    if folder_classes != class_names:
        raise ValueError(
            f"Checkpoint class names {class_names} do not match "
            f"folder class names {folder_classes}."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── inference ────────────────────────────────────────────────────
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    total = len(all_labels_np)

    # ── confusion matrix (manual — avoids sklearn import at module level) ─
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(all_labels_np, all_preds_np):
        cm[true_label][pred_label] += 1

    # ── derive metrics ───────────────────────────────────────────────
    accuracy = float(np.trace(cm)) / total if total else 0.0

    precision_per_class: dict[str, float] = {}
    recall_per_class: dict[str, float] = {}
    f1_per_class: dict[str, float] = {}
    specificity_per_class: dict[str, float] = {}
    support_per_class: dict[str, int] = {}

    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        precision_per_class[name] = round(prec, 4)
        recall_per_class[name] = round(rec, 4)
        f1_per_class[name] = round(f1, 4)
        specificity_per_class[name] = round(spec, 4)
        support_per_class[name] = int(cm[idx, :].sum())

    # Macro averages
    macro_precision = round(float(np.mean(list(precision_per_class.values()))), 4)
    macro_recall = round(float(np.mean(list(recall_per_class.values()))), 4)
    macro_f1 = round(float(np.mean(list(f1_per_class.values()))), 4)

    return {
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "precision": precision_per_class,
            "recall": recall_per_class,
            "f1_score": f1_per_class,
            "specificity": specificity_per_class,
            "support": support_per_class,
        },
        "total_samples": total,
    }
