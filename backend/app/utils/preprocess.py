# backend/app/utils/preprocess.py
from PIL import Image
from torchvision.transforms import functional as F

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return F.to_tensor(image)
