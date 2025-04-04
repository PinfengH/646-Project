# stage2_matcher.py

import os
import torch
import clip
from PIL import Image
from torchvision import transforms
from utils.datasets import RefExpDataset
from utils.vis import draw_prediction

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Create dataset
refexp_json = "data/refcoco_mini.json"
proposals_json = "outputs/proposals.json"
regions_dir = "outputs/regions"
dataset = RefExpDataset(refexp_json, proposals_json, regions_dir, transform=preprocess)

# Output directory for visualizations
save_dir = "outputs/vis"
os.makedirs(save_dir, exist_ok=True)

num_matched = 0
num_total = 0

for i in range(len(dataset)):
    sample = dataset[i]
    sentence = sample['sentence']
    regions = sample['regions']
    region_bboxes = sample['region_bboxes']
    gt_bbox = sample['gt_bbox']
    file_name = sample['file_name']

    if len(regions) == 0:
        continue

    print(f"\n[{i}] {file_name}\nSentence: \"{sentence}\"")

    # Encode text
    text_token = clip.tokenize([sentence]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_token)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    # Encode region images
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

    # Compute cosine similarity
    sims = (region_feats @ text_feat.T).squeeze()  # (N,)
    best_idx = sims.argmax().item()
    best_bbox = region_bboxes[best_idx]
    best_score = sims[best_idx].item()

    print(f"Predicted region index: {best_idx}, Score: {best_score:.4f}")

    # Convert GT bbox to xyxy
    x, y, w, h = gt_bbox
    gt_bbox_xyxy = [x, y, x + w, y + h]

    # Draw and save
    image_path = os.path.join("data/images/train2014", file_name)
    save_path = os.path.join(save_dir, f"{sample['image_id']}_{i}.png")
    draw_prediction(image_path, gt_bbox, best_bbox, save_path=save_path, title=sentence)

    num_total += 1
    # Optional: check IoU or correctness (not implemented yet)

print(f"\n✅ Processed {num_total} samples with region proposals.")
