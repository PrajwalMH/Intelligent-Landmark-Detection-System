import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T

class LandmarkDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filename']).convert("RGB")

        boxes = torch.tensor([[row['x1'], row['y1'], row['x2'], row['y2']]], dtype=torch.float32)
        labels = torch.tensor([row['class']], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.df)
