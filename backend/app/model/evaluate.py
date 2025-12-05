import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from backend.app.model.model import LandmarkModel
from backend.app.model.utils import load_image, preprocess_image

def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    return df

def evaluate_model(model, annotations):
    y_true_cls = []
    y_pred_cls = []
    bbox_errors = []
    iou_scores = []

    for _, row in annotations.iterrows():
        image_path = row['filename']
        true_class = row['class']
        true_bbox = [row['x1'], row['y1'], row['x2'], row['y2']]

        image = load_image(image_path)
        input_tensor = preprocess_image(image).unsqueeze(0)

        with torch.no_grad():
            pred_class, pred_bbox = model(input_tensor)

        y_true_cls.append(true_class)
        y_pred_cls.append(pred_class.item())

        pred_bbox_list = pred_bbox.squeeze().tolist()
        error = torch.nn.functional.mse_loss(pred_bbox.squeeze(), torch.tensor(true_bbox, dtype=torch.float))
        bbox_errors.append(error.item())

        iou = compute_iou(pred_bbox_list, true_bbox)
        iou_scores.append(iou)

    acc = accuracy_score(y_true_cls, y_pred_cls)
    avg_bbox_error = sum(bbox_errors) / len(bbox_errors)
    avg_iou = sum(iou_scores) / len(iou_scores)

    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Average Bounding Box Error (L2): {avg_bbox_error:.2f}")
    print(f"Average IoU: {avg_iou:.4f}")


def compute_iou(boxA, boxB):
    # boxA and boxB are [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


if __name__ == "__main__":
    model = LandmarkModel()
    model.load_state_dict(torch.load("backend/app/model/model.pth"))
    model.eval()

    annotations = load_annotations("backend/app/model/annotations.csv")
    evaluate_model(model, annotations)
