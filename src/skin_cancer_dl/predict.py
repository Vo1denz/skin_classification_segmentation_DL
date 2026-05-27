from __future__ import annotations

import argparse

from .inference import SkinCancerPipeline
from .utils import ensure_dir, read_image_rgb, save_json, write_image_rgb, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmentation, classification, and Grad-CAM for one image.")
    parser.add_argument("--image", required=True, help="Input dermoscopy image.")
    parser.add_argument("--classifier-checkpoint", required=True, help="Trained classifier checkpoint.")
    parser.add_argument("--segmentation-checkpoint", default=None, help="Optional trained U-Net checkpoint.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-xai", action="store_true", help="Disable Grad-CAM generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    image = read_image_rgb(args.image)
    pipeline = SkinCancerPipeline.from_checkpoints(
        classifier_checkpoint=args.classifier_checkpoint,
        segmentation_checkpoint=args.segmentation_checkpoint,
    )
    result = pipeline.predict(image, top_k=args.top_k, explain=not args.no_xai)

    if result["mask"] is not None:
        write_mask(output_dir / "lesion_mask.png", result["mask"])
    write_image_rgb(output_dir / "segmented_lesion.png", result["segmented_image"])

    if result["gradcam_overlay"] is not None:
        write_image_rgb(output_dir / "gradcam_overlay.png", result["gradcam_overlay"])

    save_json(output_dir / "prediction.json", result["prediction"])
    print(result["prediction"])


if __name__ == "__main__":
    main()
