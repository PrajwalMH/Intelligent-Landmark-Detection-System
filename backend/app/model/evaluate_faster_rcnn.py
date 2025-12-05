import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from backend.app.model.faster_rcnn import get_faster_rcnn_model
from backend.app.model.evaluate import compute_iou
from PIL import Image
from torchvision import transforms

CLASS_MAP = {
    "Eiffel_Tower": 1,
    "Taj_Mahal": 2
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transforms.ToTensor()(image)

def evaluate_faster_rcnn(model, annotations_csv):
    df = pd.read_csv(annotations_csv)
    y_true_cls = []
    y_pred_cls = []
    iou_scores = []

    for _, row in df.iterrows():
        image_path = os.path.join("dataset", row["filename"])
        true_class = CLASS_MAP[row["class"]]
        true_bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]

        image_tensor = load_image(image_path).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)[0]

        if len(outputs["boxes"]) == 0:
            continue  # Skip if no prediction

        pred_bbox = outputs["boxes"][0].cpu().tolist()
        pred_class = outputs["labels"][0].item()

        y_true_cls.append(true_class)
        y_pred_cls.append(pred_class)

        iou = compute_iou(pred_bbox, true_bbox)
        iou_scores.append(iou)

    acc = accuracy_score(y_true_cls, y_pred_cls)
    avg_iou = sum(iou_scores) / len(iou_scores)

    print(f"Faster R-CNN Accuracy: {acc:.4f}")
    print(f"Faster R-CNN Average IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_faster_rcnn_model(num_classes=3)  # 2 classes + background
    model.load_state_dict(torch.load("faster_rcnn.pth", map_location=device))
    model.to(device)
    model.eval()

    evaluate_faster_rcnn(model, "dataset/annotations.csv")
