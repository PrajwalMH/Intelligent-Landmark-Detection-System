import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

# ------------------------------------------------------------
# 1) Project paths so we can import backend.app.model
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../landmark-detection-main/demo
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) # .../landmark-detection-main

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.model.custom_model import SimpleLandmarkDetector  # noqa: E402

# ------------------------------------------------------------
# 2) Paths and configuration
# ------------------------------------------------------------
IMAGE_DIR = os.path.join(PROJECT_ROOT, "dataset")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "backend", "app", "model", "custom_model_weights.pth")

# You currently have only Eiffel Tower
# You currently have only Eiffel Tower
ANNOTATION_SAMPLE = [
    "Eiffel_Tower/001.jpg",
    "Eiffel_Tower/002.jpg",
    "Eiffel_Tower/003.jpg",
    "Eiffel_Tower/004.jpg",
    "Eiffel_Tower/005.jpeg",
    "Eiffel_Tower/006.jpg",
    "Eiffel_Tower/007.jpg",
    "Eiffel_Tower/008.jpg",
    "Eiffel_Tower/009.jpg",
    "Eiffel_Tower/010.jpg",
    "Eiffel_Tower/011.jpg",
    "Eiffel_Tower/012.jpg",
    "Eiffel_Tower/013.jpg",
    "Eiffel_Tower/014.jpg",
    "Eiffel_Tower/015.jpg",
    "Eiffel_Tower/016.jpg",
    "Eiffel_Tower/017.jpg",
    "Eiffel_Tower/018.jpg",
    "Eiffel_Tower/019.jpg",
]


# Only one class for now
CLASS_NAMES = ["Eiffel_Tower"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# 3) Model loading
# ------------------------------------------------------------
def load_model():
    model = SimpleLandmarkDetector(num_classes=len(CLASS_NAMES)).to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from: {WEIGHTS_PATH}")
        state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("Warning: custom_model_weights.pth not found.")
        print("Model will run with randomly initialized weights. Predictions will not be meaningful.")

    model.eval()
    return model

# ------------------------------------------------------------
# 4) Transform and helper
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def run_inference(model, img_path: str):
    if not os.path.exists(img_path):
        print(f"Image not found, skipping: {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # same transform you used for training
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        class_logits, bbox_preds = model(img_tensor)
        class_idx = torch.argmax(class_logits, dim=1).item()
        bbox_norm = bbox_preds.squeeze().cpu().numpy()

    print(f"Raw bbox (normalized) from model: {bbox_norm}")

    # Convert from normalized [0,1] to pixel coords
    x1 = float(bbox_norm[0]) * w
    y1 = float(bbox_norm[1]) * h
    x2 = float(bbox_norm[2]) * w
    y2 = float(bbox_norm[3]) * h

    # Sort coordinates so left < right, top < bottom
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    # Clamp to image bounds
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    # Check for degenerate / tiny boxes
    if x2 - x1 < 2 or y2 - y1 < 2:
        print("Warning: predicted bounding box is invalid or too small; "
              "showing image without rectangle.")
        image.show()
        return

    print(f"Using pixel bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, max(0, y1 - 20)), CLASS_NAMES[class_idx], fill="red")

    image.show()


# ------------------------------------------------------------
# 5) Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using dataset dir: {IMAGE_DIR}")
    print(f"Using weights: {WEIGHTS_PATH}")
    print()

    model = load_model()

    for filename in ANNOTATION_SAMPLE:
        img_path = os.path.join(IMAGE_DIR, filename)
        print(f"Running inference on: {filename}")
        run_inference(model, img_path)
