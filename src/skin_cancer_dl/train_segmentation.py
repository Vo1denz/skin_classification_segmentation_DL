from __future__ import annotations

import argparse

import torch
from torch import nn

from .datasets import make_segmentation_loaders
from .losses import BCEDiceLoss
from .models import build_unet
from .utils import AverageMeter, dice_score_from_logits, get_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net for skin lesion segmentation.")
    parser.add_argument("--image-dir", required=True, help="Folder containing dermoscopy images.")
    parser.add_argument("--mask-dir", required=True, help="Folder containing lesion masks with matching file stems.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/unet_best.pt")
    return parser.parse_args()


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
    dices = AverageMeter()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_score_from_logits(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        losses.update(float(loss.detach().cpu()), batch_size)
        dices.update(float(dice.detach().cpu()), batch_size)

    return losses.average, dices.average


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader = make_segmentation_loaders(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_unet(base_channels=args.base_channels).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(
                args.output,
                {
                    "task": "segmentation",
                    "architecture": "unet",
                    "model_state": model.state_dict(),
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "epoch": epoch,
                    "val_dice": best_dice,
                },
            )
            print(f"saved={args.output} val_dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
