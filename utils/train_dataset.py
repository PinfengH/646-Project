# utils/train_dataset.py

import os
import json
from PIL import Image
from torch.utils.data import Dataset

class RefExpTrainDataset(Dataset):
    def __init__(self, 
                 refexp_json_path,           # e.g. data/refcoco_mini.json
                 proposals_json_path,       # e.g. outputs/proposals.json
                 region_root_dir,           # e.g. outputs/regions/
                 transform=None):

        with open(refexp_json_path, 'r') as f:
            all_samples = json.load(f)

        with open(proposals_json_path, 'r') as f:
            self.proposals = json.load(f)

        self.region_root = region_root_dir
        self.transform = transform
        self.samples = []

        for item in all_samples:
            file_name = item['file_name']
            image_id = file_name.split('_')[-1].split('.')[0]
            if image_id not in self.proposals:
                continue

            region_dir = os.path.join(region_root_dir, image_id)
            if not os.path.isdir(region_dir):
                continue

            boxes = self.proposals[image_id]
            gt_bbox = item['bbox']  # [x, y, w, h]
            gt_xyxy = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]

            # Find best match index
            best_iou = 0
            best_idx = -1
            for i, box in enumerate(boxes):
                box_xyxy = box['box']
                iou = self.compute_iou(gt_xyxy, box_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx == -1:
                continue

            self.samples.append({
                'sentence': item['sentence'],
                'image_id': image_id,
                'file_name': file_name,
                'target_index': best_idx,
                'region_count': len(boxes)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        sentence = sample['sentence']
        target_index = sample['target_index']

        region_dir = os.path.join(self.region_root, image_id)
        region_images = []
        for i in range(sample['region_count']):
            region_path = os.path.join(region_dir, f'region_{i}.jpg')
            if not os.path.exists(region_path):
                continue
            image = Image.open(region_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            region_images.append(image)

        return {
            'text': sentence,
            'region_images': region_images,
            'target_index': target_index,
            'image_id': image_id,
            'file_name': sample['file_name']
        }

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - interArea
        return interArea / union if union != 0 else 0.0
