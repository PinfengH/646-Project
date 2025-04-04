# vis_clip.py

import os
import json
from utils.vis import draw_prediction

# Load predictions
pred_path = "outputs/predictions_clip.json"
with open(pred_path, "r") as f:
    predictions = json.load(f)

save_dir = "outputs/vis_clip"
os.makedirs(save_dir, exist_ok=True)

for i, item in enumerate(predictions):
    image_id = item['image_id']
    file_name = item['file_name']
    sentence = item['sentence']
    gt_bbox = item['gt_bbox']
    pred_bbox = item['pred_bbox']

    image_path = os.path.join("data/images/train2014", file_name)
    save_path = os.path.join(save_dir, f"{image_id}_{i}.png")

    draw_prediction(
        image_path=image_path,
        gt_bbox=gt_bbox,
        pred_bbox=pred_bbox,
        save_path=save_path,
        title=sentence
    )

print(f"Saved visualizations to {save_dir} ({len(predictions)} samples)")
