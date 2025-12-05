# backend/app/model/baseline_model.py
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def run_baseline_model(image_tensor):
    with torch.no_grad():
        predictions = model([image_tensor])[0]

    boxes = predictions["boxes"].tolist()
    labels = [str(label.item()) for label in predictions["labels"]]
    scores = [float(score) for score in predictions["scores"]]

    return {"boxes": boxes, "labels": labels, "scores": scores}
