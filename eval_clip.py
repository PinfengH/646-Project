# eval_clip.py

import json
import os

def compute_iou(boxA, boxB):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

# Load predictions
pred_path = "outputs/predictions_clip.json"
with open(pred_path, "r") as f:
    predictions = json.load(f)

num_total = len(predictions)
num_correct = 0
ious = []

for item in predictions:
    gt = item['gt_bbox']
    pred = item['pred_bbox']
    iou = compute_iou(gt, pred)
    ious.append(iou)
    if iou >= 0.5:
        num_correct += 1

acc = num_correct / num_total
mean_iou = sum(ious) / len(ious)

print(f"\nEvaluation Results:")
print(f"Total Samples: {num_total}")
print(f"Correct Matches (IoU >= 0.5): {num_correct}")
print(f"Accuracy: {acc:.3f}")
print(f"Mean IoU: {mean_iou:.3f}")
