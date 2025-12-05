import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from PIL import Image
from torchvision import transforms

from backend.app.model.custom_model import SimpleLandmarkDetector
from backend.app.model.faster_rcnn import get_faster_rcnn_model

# Class mapping
CLASS_MAP = {
    "Eiffel_Tower": 0,
    "Taj_Mahal": 1
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate(model, annotations_csv, is_faster_rcnn=False):
    df = pd.read_csv(annotations_csv)
    y_true_cls, y_pred_cls, iou_scores = [], [], []

    for _, row in df.iterrows():
        image_path = os.path.join("dataset", row["filename"])
        true_class = CLASS_MAP[row["class"]]
        true_bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            if is_faster_rcnn:
                outputs = model(image_tensor)[0]
                if len(outputs["boxes"]) == 0:
                    continue
                pred_bbox = outputs["boxes"][0].cpu().tolist()
                pred_class = outputs["labels"][0].item() - 1  # Adjust for background class
            else:
                logits, bbox = model(image_tensor)
                pred_class = torch.argmax(logits, dim=1).item()
                pred_bbox = bbox.squeeze().cpu().tolist()

        y_true_cls.append(true_class)
        y_pred_cls.append(pred_class)
        iou_scores.append(compute_iou(pred_bbox, true_bbox))

    acc = accuracy_score(y_true_cls, y_pred_cls)
    avg_iou = sum(iou_scores) / len(iou_scores)
    return acc, avg_iou

if __name__ == "__main__":
    # Load custom model
    custom_model = SimpleLandmarkDetector(num_classes=2).to(device)
    custom_model.load_state_dict(torch.load("backend/app/model/custom_model_weights.pth", map_location=device))
    custom_model.eval()

    # Load Faster R-CNN
    faster_model = get_faster_rcnn_model(num_classes=3).to(device)
    # faster_model.load_state_dict(torch.load("faster_rcnn.pth", map_location=device))
    faster_model.eval()

    # Evaluate both
    annotations_csv = "dataset/annotations.csv"
    acc_custom, iou_custom = evaluate(custom_model, annotations_csv, is_faster_rcnn=False)
    acc_faster, iou_faster = evaluate(faster_model, annotations_csv, is_faster_rcnn=True)

    print(f"\n📊 Evaluation Results:")
    print(f"🔵 Custom Model → Accuracy: {acc_custom:.4f}, Avg IoU: {iou_custom:.4f}")
    print(f"🔴 Faster R-CNN → Accuracy: {acc_faster:.4f}, Avg IoU: {iou_faster:.4f}")

    if acc_custom >= acc_faster and iou_custom >= iou_faster:
        print("\n✅ Your custom model performs better. It will be used going forward.")
    else:
        print("\n🚀 Faster R-CNN performs better. Consider improving your custom model.")
