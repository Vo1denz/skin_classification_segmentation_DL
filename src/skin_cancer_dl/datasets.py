from __future__ import annotations

import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .utils import image_to_tensor, list_images, mask_to_tensor, read_image_rgb, read_mask


class LesionSegmentationDataset(Dataset):
    def __init__(self, image_dir: str | Path, mask_dir: str | Path, image_size: int = 256, augment: bool = False) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augment = augment

        images = list_images(self.image_dir)
        masks = {p.stem: p for p in list_images(self.mask_dir)}
        self.samples = [(image, masks[image.stem]) for image in images if image.stem in masks]

        if not self.samples:
            raise ValueError(
                f"No matching image/mask pairs found in {self.image_dir} and {self.mask_dir}. "
                "The image and mask file stems must match."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = read_image_rgb(image_path)
        mask = read_mask(mask_path)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.2:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)

        return image_to_tensor(image), mask_to_tensor(mask)


def make_segmentation_loaders(
    image_dir: str | Path,
    mask_dir: str | Path,
    image_size: int,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    base_dataset = LesionSegmentationDataset(image_dir, mask_dir, image_size=image_size, augment=False)
    val_count = max(1, int(len(base_dataset) * val_split))
    train_count = len(base_dataset) - val_count
    if train_count <= 0:
        raise ValueError("Need at least two segmentation samples when using a validation split.")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(base_dataset), generator=generator).tolist()
    val_indices = permutation[:val_count]
    train_indices = permutation[val_count:]
    train_ds = Subset(LesionSegmentationDataset(image_dir, mask_dir, image_size=image_size, augment=True), train_indices)
    val_ds = Subset(LesionSegmentationDataset(image_dir, mask_dir, image_size=image_size, augment=False), val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def classification_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_classification_loaders(
    train_dir: str | Path,
    val_dir: str | Path | None,
    image_size: int,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dir = Path(train_dir)
    train_transform = classification_transforms(image_size, train=True)
    val_transform = classification_transforms(image_size, train=False)

    if val_dir:
        train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
        val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
        if train_ds.classes != val_ds.classes:
            raise ValueError("Training and validation class folders must have the same class names.")
        class_names = train_ds.classes
    else:
        base_for_classes = datasets.ImageFolder(train_dir)
        indices = list(range(len(base_for_classes)))
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        val_count = max(1, int(len(indices) * val_split))
        train_indices = permutation[val_count:]
        val_indices = permutation[:val_count]
        if not train_indices:
            raise ValueError("Need at least two classification samples when using a validation split.")
        train_ds = Subset(datasets.ImageFolder(train_dir, transform=train_transform), train_indices)
        val_ds = Subset(datasets.ImageFolder(train_dir, transform=val_transform), val_indices)
        class_names = base_for_classes.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, class_names
