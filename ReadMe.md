# Landmark Detection

This project performs visual landmark recognition through deep learning-based object detection. It compares a custom-trained model against a Faster R-CNN baseline, evaluates their performance, and visually annotates predictions with bounding boxes.


## Project Overview

- Trains a **custom detector** from scratch and compares its performance to **Faster R-CNN**.
- Handles **normalized** (custom) and **absolute pixel** (Faster R-CNN) bounding boxes.
- Supports COCO-style datasets and VOC format.
- Generates annotated images for comparison.
- Includes reproducibility controls with seed setup and deterministic data handling.

---




# Installation

Make sure you have **Python 3.9+** installed, then set up the project:

```bash
python -m venv venv
source venv/bin/activate     # On Windows use: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Dataset Format

Default format: **COCO JSON**  
Optional support: **Pascal VOC XML**

Expected layout for each split (`train`, `val`, `test`):

```text
data/
├─ train/
│  ├─ images/
│  └─ annotations.json
├─ val/
│  ├─ images/
│  └─ annotations.json
```

Define class labels either inside the annotation file or in `configs/classes.txt`.

# Model Configurations

Custom model config snippet (`configs/custom.yaml`):

```yaml
device: "cuda"
seed: 42

data:
  format: "coco"
  train_dir: "data/train/images"
  val_dir: "data/val/images"
  train_ann: "data/train/annotations.json"
  val_ann: "data/val/annotations.json"
  classes: ["Taj_Mahal", "Monument", "Temple"]

model:
  name: "custom_detector_v1"
  num_classes: 3
  image_size: [640, 640]
  anchor_scales: [32, 64, 128, 256, 512]

train:
  epochs: 50
  batch_size: 8
  lr: 0.0005
  amp: true

eval:
  iou_thresholds: [0.5, 0.75]
  score_threshold: 0.25
```

# Training & Evaluation

*Train the custom detector*

```bash
python -m train_custom_model.py --config configs/custom.yaml
```

*Evaluate model performance*

```bash
python -m evaluate_models.py --config configs/custom.yaml   --checkpoint outputs/checkpoints/custom_best.pt
```

*Compare both models*

```bash
python compare_models.py   --custom_ckpt outputs/checkpoints/custom_best.pt   --baseline faster_rcnn
```
*Run Output*

```bash
python run_demo.py   
```

# 📈 Evaluation Results

## Overall Comparison

| Metric | Custom Model | Faster R-CNN |
| ------ | ------------ | ------------ |
| mAP@0.50 | 0.71 | 0.67 |
| mAP@0.50:0.95 | 0.44 | 0.41 |
| Precision | 0.83 | 0.80 |
| Recall | 0.78 | 0.74 |
| Inference FPS | 42 | 18 |
| Parameters | 24 M | 41 M |

## Per Class AP (mAP@0.50)

| Class | Custom Model | Faster R-CNN |
| ----- | ------------ | ------------ |
| Taj Mahal | 0.78 | 0.73 |
| Eiffle tower | 0.70 | 0.66 |

# Architecture Insights

* **Custom Model**: Modular design with scalable anchor based detection  
* **Faster R-CNN**: Standard implementation from Torchvision

Bounding box formats:

* **Normalized boxes** (custom): `[x1, y1, x2, y2]` scaled by image dimensions  
* **Pixel boxes** (Faster R-CNN): `[x1, y1, x2, y2]` in absolute pixels

