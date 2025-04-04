# predict_clip.py

import os
import json
import torch
import clip
from utils.datasets import RefExpDataset

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load dataset
refexp_json = "data/refcoco_mini.json"
proposals_json = "outputs/proposals.json"
regions_dir = "outputs/regions"
dataset = RefExpDataset(refexp_json, proposals_json, regions_dir, transform=preprocess)

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

    # Encode text
    text_token = clip.tokenize([sentence]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_token)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    # Encode regions
    region_feats = []
    with torch.no_grad():
        for region in regions:
            image_input = region.unsqueeze(0).to(device)
            feat = model.encode_image(image_input)
            feat /= feat.norm(dim=-1, keepdim=True)
            region_feats.append(feat)

    if not region_feats:
        continue

    region_feats = torch.cat(region_feats, dim=0)

    # Similarity
    sims = (region_feats @ text_feat.T).squeeze()
    best_idx = sims.argmax().item()
    if sims.dim() == 0:
        best_score = sims.item()
    else:
        best_score = sims[best_idx].item()

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
save_path = "outputs/predictions_clip.json"
os.makedirs("outputs", exist_ok=True)
with open(save_path, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"Saved predictions to {save_path} ({len(predictions)} samples)")