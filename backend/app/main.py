from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torchvision.transforms as T
import io
import torch


from backend.app.model.custom_model import run_custom_model
from backend.app.model.faster_rcnn import get_faster_rcnn_model

app = FastAPI()
device = "cpu"

faster_model = get_faster_rcnn_model(num_classes=3).to(device)
faster_model.eval()

@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = T.ToTensor()(image).unsqueeze(0)

    # Custom model
    custom_result = run_custom_model(tensor[0])

    # Faster R-CNN
    with torch.no_grad():
        outputs = faster_model(tensor)[0]
        if len(outputs["boxes"]) > 0:
            box = outputs["boxes"][0].tolist()
            label = f"Class_{outputs['labels'][0].item()}"
            score = outputs["scores"][0].item()
            faster_result = {
                "boxes": [box],
                "labels": [label],
                "scores": [score]
            }
        else:
            faster_result = {
                "boxes": [[0, 0, 0, 0]],
                "labels": ["None"],
                "scores": [0.0]
            }

    return JSONResponse(content={
        "custom": custom_result,
        "faster_rcnn": faster_result
    })
