import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from backend.app.data.loader import LandmarkDataset
from backend.app.model.custom_model import SimpleLandmarkDetector

# Config
NUM_CLASSES = 2
EPOCHS = 20
BATCH_SIZE = 4
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = LandmarkDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = SimpleLandmarkDetector(num_classes=NUM_CLASSES).to(DEVICE)
criterion_cls = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels, bboxes in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        bboxes = bboxes.to(DEVICE)

        optimizer.zero_grad()
        class_logits, bbox_preds = model(images)
        loss_cls = criterion_cls(class_logits, labels)
        loss_bbox = criterion_bbox(bbox_preds, bboxes)
        loss = loss_cls + loss_bbox
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# Save weights
torch.save(model.state_dict(), "./backend/app/model/custom_model_weights.pth")
print("✅ Model trained and weights saved.")
