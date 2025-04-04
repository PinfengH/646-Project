import os
import json
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from utils.image_utils import crop_regions


# Load pre-trained Faster R-CNN model
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model


# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image), image


# Generate region proposals
def get_region_proposals(model, image_tensor, score_thresh=0.7, top_k=20):
    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs['boxes']
    scores = outputs['scores']

    # Filter by score threshold
    mask = scores > score_thresh
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]

    # Keep top_k
    if filtered_boxes.size(0) > top_k:
        filtered_boxes = filtered_boxes[:top_k]
        filtered_scores = filtered_scores[:top_k]

    return filtered_boxes.cpu(), filtered_scores.cpu()


# Main function: batch process images
def main():
    image_dir = "data/images/train2014"
    save_root = "outputs/regions"
    os.makedirs(save_root, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])[:5000]  # Only process first 5000 images

    print(f"Found {len(image_files)} images.")
    model = load_model()
    all_boxes = defaultdict(list)

    for idx, image_file in enumerate(image_files):
        image_id = image_file.split("_")[-1].split(".")[0]  # e.g. 000000123456
        image_path = os.path.join(image_dir, image_file)
        print(f"[{idx+1}/{len(image_files)}] Processing {image_file}...")

        image_tensor, raw_image = load_image(image_path)
        boxes, scores = get_region_proposals(model, image_tensor, score_thresh=0.7)

        crops = crop_regions(raw_image, boxes)

        # Save regions
        save_dir = os.path.join(save_root, image_id)
        os.makedirs(save_dir, exist_ok=True)
        for i, crop in enumerate(crops):
            save_path = os.path.join(save_dir, f"region_{i}.jpg")
            crop.save(save_path)

        # Save box coordinates
        for box, score in zip(boxes.tolist(), scores.tolist()):
            all_boxes[image_id].append({
                "box": [round(v, 2) for v in box],
                "score": round(score, 4)
            })

    # Save all proposals to JSON
    with open("outputs/proposals.json", "w") as f:
        json.dump(all_boxes, f, indent=2)
    print(f"\nSaved proposals to outputs/proposals.json")


if __name__ == "__main__":
    main()

