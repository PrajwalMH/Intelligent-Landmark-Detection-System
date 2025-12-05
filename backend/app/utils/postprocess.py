# backend/app/utils/postprocess.py
def format_output(outputs):
    result = []
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        result.append({
            "label": label,
            "confidence": round(score, 2),
            "box": [round(coord, 2) for coord in box]
        })
    return {"detections": result}
