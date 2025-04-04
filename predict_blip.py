# predict_blip.py

import os
import json
import torch
from torchvision import transforms
from PIL import Image
from utils.datasets import RefExpDataset
from transformers import BlipProcessor, BlipModel

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
model = BlipModel.from_pretrained("Salesforce/blip-itm-base-coco").to(device)
model.eval()

# Load dataset
refexp_json = "data/refcoco_mini.json"
proposals_json = "outputs/proposals.json"
regions_dir = "outputs/regions"
dataset = RefExpDataset(refexp_json, proposals_json, regions_dir)

predictions = []

for i in range(len(dataset)):
    sample = dataset[i]
    sentence = sample['sentence']
    regions = sample['regions']
    region_bboxes = sample['region_bboxes']
    gt_bbox = sample['gt_bbox']
    file_name = sample['file_name']
    image_id = sample['image_id']

    if len(regions) == 0:
        continue

    scores = []
    for region in regions:
        inputs = processor(images=region, text=sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.itm_score[:, 1].item()  # BLIP outputs score for matching class
            scores.append(score)

    best_idx = int(torch.tensor(scores).argmax().item())
    best_score = float(scores[best_idx])
    best_bbox = region_bboxes[best_idx]

    # Convert GT bbox [x, y, w, h] → [x1, y1, x2, y2]
    x, y, w, h = gt_bbox
    gt_bbox_xyxy = [x, y, x + w, y + h]

    predictions.append({
        "image_id": image_id,
        "file_name": file_name,
        "sentence": sentence,
        "pred_index": best_idx,
        "score": round(best_score, 4),
        "pred_bbox": [round(v, 2) for v in best_bbox],
        "gt_bbox": [round(x, 2) for x in gt_bbox_xyxy]
    })

# Save to JSON
save_path = "outputs/predictions_blip.json"
os.makedirs("outputs", exist_ok=True)
with open(save_path, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"Saved BLIP predictions to {save_path} ({len(predictions)} samples)")
