from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import Subset

from .datasets import make_classification_loaders
from .models import build_classifier, freeze_backbone
from .utils import AverageMeter, accuracy_from_logits, get_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lesion classifier with transfer learning.")
    parser.add_argument("--train-dir", required=True, help="ImageFolder style training directory.")
    parser.add_argument("--val-dir", default=None, help="Optional ImageFolder style validation directory.")
    parser.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v2", "resnet18"])
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained ImageNet weights.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the final classifier head.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weights.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/classifier_best.pt")
    return parser.parse_args()


def dataset_targets(dataset: torch.utils.data.Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base_targets = dataset_targets(dataset.dataset)
        return [base_targets[index] for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    raise TypeError("Cannot read targets from this dataset type.")


def class_weights(dataset: torch.utils.data.Dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    targets = torch.tensor(dataset_targets(dataset), dtype=torch.long)
    counts = torch.bincount(targets, minlength=num_classes).float()
    weights = counts.sum() / counts.clamp(min=1.0)
    weights = weights / weights.mean()
    return weights.to(device)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = AverageMeter()
    accuracies = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            accuracy = accuracy_from_logits(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        losses.update(float(loss.detach().cpu()), batch_size)
        accuracies.update(float(accuracy.detach().cpu()), batch_size)

    return losses.average, accuracies.average


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, class_names = make_classification_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    spec = build_classifier(args.model, num_classes=len(class_names), pretrained=args.pretrained)
    model = spec.model.to(device)
    if args.freeze_backbone:
        freeze_backbone(model, args.model)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    weights = None if args.no_class_weights else class_weights(train_loader.dataset, len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                args.output,
                {
                    "task": "classification",
                    "model_name": args.model,
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": len(class_names),
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_accuracy": best_accuracy,
                },
            )
            print(f"saved={args.output} val_acc={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
