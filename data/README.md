# Dataset Folder

Place datasets here.

## Segmentation

```text
data/segmentation/images/<image_id>.jpg
data/segmentation/masks/<image_id>.png
```

The mask should be a binary or grayscale image where lesion pixels are bright.

## Classification

```text
data/classification/train/<class_name>/*.jpg
data/classification/val/<class_name>/*.jpg
```

Example classes:

```text
Melanoma
Basal_Cell_Carcinoma
Nevus
Benign_Keratosis
```
