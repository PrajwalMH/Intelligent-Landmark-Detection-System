import torch
from torch.utils.data import DataLoader
from backend.app.model.faster_rcnn import get_faster_rcnn_model
from backend.app.model.faster_dataset import LandmarkDetectionDataset
import torchvision.transforms as T

def train():
    num_classes = 6  # 5 landmarks + background
    model = get_faster_rcnn_model(num_classes)

    dataset = LandmarkDetectionDataset("annotations.csv", transforms=T.ToTensor())
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    model.train()
    for epoch in range(5):
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {losses.item():.4f}")

    torch.save(model.state_dict(), "faster_rcnn.pth")

if __name__ == "__main__":
    train()
