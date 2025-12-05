# backend/app/api.py
import io
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as T

from backend.app.model.custom_model import SimpleLandmarkDetector

# ---------------------- FastAPI app ----------------------
app = FastAPI()

# Allow requests from React dev server (http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Model loading ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "app", "model", "custom_model_weights.pth")

# Adjust num_classes if you have more
NUM_CLASSES = 1
model = SimpleLandmarkDetector(num_classes=NUM_CLASSES).to(DEVICE)

state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# ---------------------- API endpoint ----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image and returns a normalized bbox in [0,1] coords.
    """
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = image.size

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        class_logits, bbox = model(x)
        bbox = bbox.squeeze().cpu().tolist()

    # Assume model outputs approx normalized coords; clamp safely
    x1, y1, x2, y2 = bbox

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    x1 = clamp01(x1)
    y1 = clamp01(y1)
    x2 = clamp01(x2)
    y2 = clamp01(y2)

    # Make sure x1<x2, y1<y2
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    return {
        "label": "Eiffel_Tower",     # or use your class index mapping
        "bbox": [x1, y1, x2, y2],    # normalized [0,1]
        "image_width": w,
        "image_height": h,
    }
