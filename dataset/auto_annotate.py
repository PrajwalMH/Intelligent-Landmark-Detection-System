import os
import csv
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Config
IMAGE_ROOT = os.path.dirname(__file__)  # Points to 'dataset' folder
CLASSES = [d for d in os.listdir(IMAGE_ROOT) if os.path.isdir(os.path.join(IMAGE_ROOT, d))]
OUTPUT_CSV = os.path.join(IMAGE_ROOT, "annotations.csv")
CONFIDENCE_THRESHOLD = 0.8

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToTensor()
])

def annotate_image(img_path):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']

    # Return top box above threshold
    for box, score in zip(boxes, scores):
        if score > CONFIDENCE_THRESHOLD:
            return [int(coord) for coord in box.tolist()]
    return None

def generate_annotations():
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x1", "y1", "x2", "y2", "class"])

        for cls in CLASSES:
            folder = os.path.join(IMAGE_ROOT, cls)
            if not os.path.isdir(folder):
                print(f"❌ Folder not found: {folder}")
                continue

            for img_name in sorted(os.listdir(folder)):
                img_path = os.path.join(folder, img_name)
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue  # Skip non-image files

                box = annotate_image(img_path)
                if box:
                    writer.writerow([f"{cls}/{img_name}", *box, cls])
                    print(f"✅ Annotated: {cls}/{img_name}")
                else:
                    print(f"⚠️ No confident box for: {cls}/{img_name}")

if __name__ == "__main__":
    generate_annotations()
