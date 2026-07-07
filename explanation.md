# Skin Cancer DL: Project Explanation

This document provides a high-level overview of the **Skin Cancer Segmentation and Classification** project. It is designed to take about 5–6 minutes to read and is perfect for reviewers looking to understand the architecture, design decisions, and functionality of each codebase component.

---

## 🏗️ 1. Project Overview & Pipeline

This project implements a complete end-to-end deep learning pipeline for analyzing dermoscopy images. The goal is to identify and classify skin lesions (e.g., Benign vs. Malignant) while providing visual explanations for its decisions.

The pipeline operates in **two main stages**, wrapped in an interactive web application:

1. **Segmentation (U-Net):** First, the system isolates the skin lesion from the background (removing healthy skin, hair, and rulers).
2. **Classification (MobileNetV2):** The isolated lesion is then passed to a classifier to predict whether it is benign or malignant.
3. **Explainable AI (Grad-CAM):** Finally, a heatmap is generated to show exactly which parts of the lesion the classifier focused on when making its decision.

---

## 📂 2. Codebase Breakdown

The core logic resides in the `src/skin_cancer_dl/` directory. Here is a summary of what each file does:

### The Core Neural Networks
* **`models.py`**
  * Defines the actual deep learning architectures.
  * Contains the **U-Net** implementation from scratch (used for segmentation).
  * Contains helper functions to load pretrained models from `torchvision` (like **MobileNetV2** or EfficientNet) and modify their final fully-connected layers to match our number of classes (Benign/Malignant).

### Data Handling
* **`datasets.py`**
  * Manages how images are loaded from disk and fed into the neural networks.
  * Defines distinct `Dataset` classes for both segmentation (loading images + masks) and classification (loading cropped lesions + labels).
  * Implements **Data Augmentation** (e.g., random flips, rotations, color jitter, and random erasing) to artificially expand the dataset and prevent the models from overfitting.

### Training the Models
* **`train_segmentation.py`**
  * The training loop for the U-Net model. It iterates over the dataset, computes the loss (using BCE-Dice), updates weights, and saves the best model based on the Validation Dice Score.
* **`train_classifier.py`**
  * The training loop for the classification model. 
  * Features a **Two-Phase Fine-Tuning** strategy: it first freezes the pretrained backbone to train the new classification head (warmup), then unfreezes everything for full fine-tuning.
  * Calculates **Class Weights** dynamically to combat the severe class imbalance (far more benign images than malignant).

* **`losses.py`**
  * Contains custom loss functions critical for medical imaging. Specifically, it implements **Dice Loss** and **BCE-Dice Loss**, which are much better suited for segmentation tasks than standard Cross-Entropy, especially when the lesion only occupies a small portion of the image.

### Inference & Explainability
* **`inference.py`**
  * The orchestrator. It defines the `SkinCancerPipeline` class which ties the whole system together. It takes an input image, runs it through U-Net, crops the output, feeds the crop to the classifier, and returns the final combined results.
* **`xai.py`**
  * Implements **Grad-CAM (Gradient-weighted Class Activation Mapping)**. 
  * It hooks into the final convolutional layers of the classifier during a backward pass to figure out which pixels strongly activated the predicted class. It then overlays this as a heatmap on the original image, providing transparency to the model's decision.
* **`evaluate.py`**
  * A dedicated module for calculating standard machine learning metrics across the validation dataset. It generates the **Confusion Matrix**, Precision, Recall, F1-Score, and Specificity, which are crucial for understanding clinical viability.

### Application & Deployment
* **`api.py`**
  * A **FastAPI** web server. It exposes a `/predict` endpoint that receives an image upload, passes it to the `SkinCancerPipeline`, and returns JSON data along with base64-encoded images (the mask, the crop, and the heatmap).
  * Also exposes an `/evaluate` endpoint to fetch the latest model performance metrics.
* **`static/` (Frontend UI)**
  * Contains `index.html`, `app.js`, and `styles.css`. This is a vanilla JavaScript, browser-based dashboard where users can upload images, see the AI's step-by-step reasoning (segmentation -> classification -> XAI), and view the model's overall performance metrics and confusion matrix.

### Utilities
* **`utils.py`**
  * A collection of helper functions used across the project (e.g., seeding random number generators for reproducibility, tensor-to-image conversion, and the `AverageMeter` class used for tracking loss during training loops).

---

## 🎯 3. Key Design Decisions to Highlight

If you are reviewing this project, pay special attention to these architectural choices:

1. **Two-Stage Approach:** By forcing the classifier to look *only* at the segmented lesion rather than the whole image, we prevent the model from "cheating" by learning background artifacts (like surgical skin markings or dark corners).
2. **Transfer Learning + Heavy Augmentation:** Medical datasets are notoriously small (e.g., only ~900 images in the starter set). To make the classification model accurate, we rely heavily on pretrained ImageNet weights and aggressive data augmentation.
3. **Addressing Class Imbalance:** Malignant cases are rare compared to benign cases. The training script automatically calculates inverse-frequency weights and applies a "minority boost" so the model is heavily penalized for missing a malignant case, optimizing for Recall.
4. **Clinical Explainability:** The inclusion of Grad-CAM ensures the system is not just a "black box." A clinician can look at the heatmap and verify if the AI is looking at the actual biological indicators of melanoma.
