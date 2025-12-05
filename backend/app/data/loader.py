import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

CLASS_MAP = {
    "Eiffel_Tower": 0,
    "Taj_Mahal": 1
}

class LandmarkDataset(Dataset):
    def __init__(self, root_dir="./dataset", annotation_file="./dataset/annotations.csv", transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = CLASS_MAP[row["class"]]
        bbox = torch.tensor([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=torch.float32)
        bbox /= 224.0  # Normalize assuming resized image

        return image, label, bbox
