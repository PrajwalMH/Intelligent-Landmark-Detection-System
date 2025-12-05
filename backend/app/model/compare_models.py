import os
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from backend.app.model.custom_model import SimpleLandmarkDetector
from backend.app.model.faster_rcnn import get_faster_rcnn_model

CLASS_NAMES = ["Eiffel_Tower", "Taj_Mahal"]

def draw_box(image, box, label, color):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = map(int, box)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1, y1 - 10), label, fill=color)
    return image

def compare_models(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0).to(device)

    # Custom model
    custom_model.eval()
    with torch.no_grad():
        logits, bbox = custom_model(tensor)
        label_idx = torch.argmax(logits, dim=1).item()
        label = CLASS_NAMES[label_idx]
        box = bbox.squeeze().cpu().numpy() * 224
        image_custom = image.copy()
        draw_box(image_custom, box, f"Custom: {label}", "blue")

    # Faster R-CNN
    faster_model.eval()
    with torch.no_grad():
        outputs = faster_model(tensor)[0]
        if len(outputs["boxes"]) > 0:
            box = outputs["boxes"][0].cpu().numpy()
            label_idx = outputs["labels"][0].item() - 1
            label = CLASS_NAMES[label_idx]
            image_faster = image.copy()
            draw_box(image_faster, box, f"FasterRCNN: {label}", "red")
        else:
            image_faster = image.copy()

    # Show side-by-side
    image_custom.show(title="Custom Model")
    image_faster.show(title="Faster R-CNN")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom_model = SimpleLandmarkDetector(num_classes=2).to(device)
    custom_model.load_state_dict(torch.load("backend/app/model/custom_model_weights.pth", map_location=device))

    faster_model = get_faster_rcnn_model(num_classes=3).to(device)
    faster_model.load_state_dict(torch.load("faster_rcnn.pth", map_location=device))

    compare_models("dataset/Taj_Mahal/002.jpg")
