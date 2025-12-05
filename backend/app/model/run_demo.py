import os
import cv2
import torch
import pandas as pd
from backend.app.model.model import LandmarkModel
from backend.app.model.utils import load_image, preprocess_image

def draw_bbox(image, bbox, label, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def run_demo_on_folder(model, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        image = load_image(image_path)
        input_tensor = preprocess_image(image).unsqueeze(0)

        with torch.no_grad():
            pred_class, pred_bbox = model(input_tensor)

        label = str(pred_class.item())
        bbox = pred_bbox.squeeze().tolist()

        image_with_bbox = draw_bbox(image.copy(), bbox, label)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image_with_bbox)

if __name__ == "__main__":
    model = LandmarkModel()
    model.load_state_dict(torch.load("backend/app/model/model.pth"))
    model.eval()

    run_demo_on_folder(model, "backend/app/model/test_images", "backend/app/model/predictions")
