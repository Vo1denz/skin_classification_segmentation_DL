from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .inference import SkinCancerPipeline
from .utils import read_image_rgb_from_bytes

app = FastAPI(title="Skin Cancer Deep Learning API", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def default_checkpoint(name: str) -> str | None:
    path = Path("checkpoints") / name
    if path.exists():
        return str(path)
    return None


def image_data_uri(image_rgb: np.ndarray, extension: str = ".png") -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(extension, image_bgr)
    if not ok:
        raise RuntimeError("Could not encode result image.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    mime = "image/png" if extension == ".png" else "image/jpeg"
    return f"data:{mime};base64,{payload}"


def mask_data_uri(mask: np.ndarray) -> str:
    mask_uint8 = np.clip(mask * 255 if mask.max() <= 1.0 else mask, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", mask_uint8)
    if not ok:
        raise RuntimeError("Could not encode mask.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


@lru_cache(maxsize=1)
def get_pipeline() -> SkinCancerPipeline:
    classifier_checkpoint = os.getenv("CLASSIFIER_CHECKPOINT") or default_checkpoint("classifier_isic2016.pt")
    segmentation_checkpoint = os.getenv("SEGMENTATION_CHECKPOINT") or default_checkpoint("unet_isic2016.pt")
    if not classifier_checkpoint:
        raise RuntimeError("Set CLASSIFIER_CHECKPOINT or train checkpoints/classifier_isic2016.pt first.")
    return SkinCancerPipeline.from_checkpoints(
        classifier_checkpoint=classifier_checkpoint,
        segmentation_checkpoint=segmentation_checkpoint,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI assets are missing.")
    return index_path.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, object]:
    classifier_checkpoint = os.getenv("CLASSIFIER_CHECKPOINT") or default_checkpoint("classifier_isic2016.pt")
    segmentation_checkpoint = os.getenv("SEGMENTATION_CHECKPOINT") or default_checkpoint("unet_isic2016.pt")
    return {
        "status": "ok",
        "classifier_checkpoint": classifier_checkpoint,
        "segmentation_checkpoint": segmentation_checkpoint,
        "ui": STATIC_DIR.exists(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        image = read_image_rgb_from_bytes(await file.read())
        result = get_pipeline().predict(image, top_k=top_k, explain=True)
        return result["prediction"]
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/visual")
async def predict_visual(file: UploadFile = File(...), top_k: int = 2) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        image = read_image_rgb_from_bytes(await file.read())
        result = get_pipeline().predict(image, top_k=top_k, explain=True)

        mask = result["mask"]
        lesion_area_percent = None
        if mask is not None:
            lesion_area_percent = float((mask > 0.5).mean() * 100.0)

        visual_payload = {
            "prediction": result["prediction"],
            "images": {
                "input": image_data_uri(image, ".jpg"),
                "lesion_mask": mask_data_uri(mask) if mask is not None else None,
                "segmented_lesion": image_data_uri(result["segmented_image"]),
                "gradcam_overlay": image_data_uri(result["gradcam_overlay"])
                if result["gradcam_overlay"] is not None
                else None,
            },
            "metadata": {
                "filename": file.filename,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "lesion_area_percent": lesion_area_percent,
                "segmentation_model": "U-Net",
                "classification_model": "MobileNetV2",
                "class_names": get_pipeline().class_names,
                "disclaimer": "Academic screening prototype only. Not for medical diagnosis.",
            },
        }
        return visual_payload
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
