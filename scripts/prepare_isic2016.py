from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ISIC 2016 data for this project.")
    parser.add_argument("--image-dir", default="data/raw/ISBI2016_ISIC_Part1_Training_Data")
    parser.add_argument("--mask-dir", default="data/raw/ISBI2016_ISIC_Part1_Training_GroundTruth")
    parser.add_argument("--labels-csv", default="data/downloads/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_labels(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            image_id, label = row[0].strip(), row[1].strip()
            if image_id and label:
                rows.append((image_id, label))
    return rows


def prepare_segmentation(image_dir: Path, mask_dir: Path, output_root: Path) -> int:
    out_images = output_root / "segmentation" / "images"
    out_masks = output_root / "segmentation" / "masks"
    count = 0

    for image_path in sorted(image_dir.glob("*.jpg")):
        image_id = image_path.stem
        mask_path = mask_dir / f"{image_id}_Segmentation.png"
        if not mask_path.exists():
            continue
        link_or_copy(image_path, out_images / image_path.name)
        link_or_copy(mask_path, out_masks / f"{image_id}.png")
        count += 1

    return count


def prepare_classification(
    image_dir: Path,
    labels: list[tuple[str, str]],
    output_root: Path,
    val_split: float,
    seed: int,
) -> dict[str, dict[str, int]]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for image_id, label in labels:
        if (image_dir / f"{image_id}.jpg").exists():
            by_class[label].append(image_id)

    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = {}

    for label, image_ids in sorted(by_class.items()):
        rng.shuffle(image_ids)
        val_count = max(1, int(len(image_ids) * val_split))
        splits = {
            "val": image_ids[:val_count],
            "train": image_ids[val_count:],
        }
        counts[label] = {}
        for split_name, split_ids in splits.items():
            class_dir = output_root / "classification" / split_name / label
            for image_id in split_ids:
                link_or_copy(image_dir / f"{image_id}.jpg", class_dir / f"{image_id}.jpg")
            counts[label][split_name] = len(split_ids)

    return counts


def main() -> int:
    args = parse_args()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    labels_csv = Path(args.labels_csv)
    output_root = Path(args.output_root)

    for path in [image_dir, mask_dir, labels_csv]:
        if not path.exists():
            print(f"Missing required path: {path}", file=sys.stderr)
            return 1

    labels = read_labels(labels_csv)
    segmentation_count = prepare_segmentation(image_dir, mask_dir, output_root)
    classification_counts = prepare_classification(
        image_dir=image_dir,
        labels=labels,
        output_root=output_root,
        val_split=args.val_split,
        seed=args.seed,
    )

    print(f"segmentation_pairs={segmentation_count}")
    for label, counts in classification_counts.items():
        print(f"class={label} train={counts['train']} val={counts['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
