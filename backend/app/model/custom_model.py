import torch
import torch.nn as nn
import torchvision.models as models


class SimpleLandmarkDetector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # 1) Pretrained ResNet-18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final FC layer & pooling – we’ll add our own
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # up to last conv
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_feats = 512  # resnet18 last conv channel size

        # 2) Heads
        self.classifier = nn.Linear(in_feats, num_classes)  # class logits
        self.bbox_regressor = nn.Linear(in_feats, 4)        # [x1, y1, x2, y2] normalized

    def forward(self, x):
        # Feature extractor
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classification branch
        class_logits = self.classifier(x)

        # Bounding box branch (sigmoid keeps coords in [0, 1])
        bbox = torch.sigmoid(self.bbox_regressor(x))

        return class_logits, bbox


# Inference function
def run_custom_model(image_tensor):
    image_tensor = T.Resize((224, 224))(image_tensor).unsqueeze(0)  # Resize and batch
    with torch.no_grad():
        class_logits, bbox = model(image_tensor)
        scores = torch.softmax(class_logits, dim=1)
        confidence, label_idx = scores.max(dim=1)
        label = f"Class_{label_idx.item()}"  # Replace with actual class name mapping
        box = bbox.squeeze().tolist()
        return {
            "boxes": [box],
            "labels": [label],
            "scores": [confidence.item()]
        }
