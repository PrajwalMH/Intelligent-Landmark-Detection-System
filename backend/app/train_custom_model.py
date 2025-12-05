import os
import sys
import time

from typing import Tuple, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ------------------------------------------------------------
# 0. Paths / project layout
#    File is: backend/app/train_custom_model.py
#    Project root: one level above "backend"
# ------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))           # .../backend/app
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, "..", ".."))  # .../landmark-detection-main
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
ANNOT_CSV = os.path.join(DATASET_DIR, "annotations.csv")
WEIGHTS_OUT = os.path.join(APP_DIR, "model", "custom_model_weights.pth")

# Make sure Python can import "backend.app.model"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.app.model.custom_model import SimpleLandmarkDetector  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# 1. Dataset
# ------------------------------------------------------------
class LandmarkDataset(Dataset):
    """
    Expects annotations.csv with columns:
    filename, class, x1, y1, x2, y2

    filename is relative to dataset root, e.g. "Eiffel_Tower/001.jpg".
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        class_to_idx: dict,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = row["filename"]
        cls_name = row["class"]

        x1, y1, x2, y2 = row[["x1", "y1", "x2", "y2"]].astype(float).tolist()

        img_path = os.path.join(self.image_root, rel_path)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # normalized [0, 1] bbox
        bbox = torch.tensor(
            [x1 / w, y1 / h, x2 / w, y2 / h], dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.class_to_idx[cls_name], dtype=torch.long)

        return image, label, bbox


# ------------------------------------------------------------
# 2. Dataloaders
# ------------------------------------------------------------
def create_dataloaders(
    batch_size: int = 4,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    if not os.path.exists(ANNOT_CSV):
        raise FileNotFoundError(
            f"annotations.csv not found at {ANNOT_CSV}\n"
            "Run auto_annotate.py from project root first."
        )

    df = pd.read_csv(ANNOT_CSV)

    required_cols = {"filename", "class", "x1", "y1", "x2", "y2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"annotations.csv is missing columns: {missing}")

    # Class mapping
    class_names = sorted(df["class"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print(f"Found {len(df)} annotated images.")
    print(f"Classes ({len(class_names)}): {class_names}")

    # Simple 80/20 split
    df_train = df.sample(frac=0.8, random_state=42)
    df_val = df.drop(df_train.index)

    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}")

    # Data augmentation for training
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.ToTensor(),
        ]
    )

    # Deterministic for validation
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_ds = LandmarkDataset(df_train, DATASET_DIR, class_to_idx, train_transform)
    val_ds = LandmarkDataset(df_val, DATASET_DIR, class_to_idx, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, class_names


# ------------------------------------------------------------
# 3. Training loop
# ------------------------------------------------------------
def train_model(
    num_epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 8,
    bbox_loss_weight: float = 1.0,
):
    print("--------------------------------------------------")
    print(f"Using device: {DEVICE}")
    print(f"Project root: {REPO_ROOT}")
    print(f"Dataset dir: {DATASET_DIR}")
    print(f"Annotations: {ANNOT_CSV}")
    print(f"Will save weights to: {WEIGHTS_OUT}")
    print("--------------------------------------------------")

    train_loader, val_loader, class_names = create_dataloaders(
        batch_size=batch_size
    )

    # Model: make sure custom_model.py has the ResNet18 + sigmoid version
    model = SimpleLandmarkDetector(num_classes=len(class_names)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cls_criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for images, labels, bboxes in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            bboxes = bboxes.to(DEVICE)

            optimizer.zero_grad()

            class_logits, bbox_preds = model(images)

            loss_cls = cls_criterion(class_logits, labels)
            loss_bbox = bbox_criterion(bbox_preds, bboxes)
            loss = loss_cls + bbox_loss_weight * loss_bbox

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ----------------- validation -----------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, bboxes in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                bboxes = bboxes.to(DEVICE)

                class_logits, bbox_preds = model(images)

                loss_cls = cls_criterion(class_logits, labels)
                loss_bbox = bbox_criterion(bbox_preds, bboxes)
                loss = loss_cls + bbox_loss_weight * loss_bbox

                val_loss += loss.item() * images.size(0)

                preds = class_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{num_epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f} "
            f"- time: {epoch_time:.1f}s"
        )

        # Save best by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f"  ✔ Saved best model weights to: {WEIGHTS_OUT}")

    print("Training completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best weights stored at: {WEIGHTS_OUT}")


if __name__ == "__main__":
    # You can tweak these if you want
    train_model(num_epochs=30, lr=1e-4, batch_size=8)
