# utils/datasets.py

import os
import json
from PIL import Image
from torch.utils.data import Dataset

class RefExpDataset(Dataset):
    def __init__(self, 
                 refexp_json_path,           # path to refcoco_mini.json
                 proposals_json_path,       # path to outputs/proposals.json
                 region_root_dir,           # path to outputs/regions/
                 transform=None):

        with open(refexp_json_path, 'r') as f:
            self.samples = json.load(f)

        with open(proposals_json_path, 'r') as f:
            self.proposals = json.load(f)

        self.region_root = region_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['file_name'].split('_')[-1].split('.')[0]  # e.g., '000000581857'
        sentence = sample['sentence']
        gt_bbox = sample['bbox']

        region_dir = os.path.join(self.region_root, image_id)
        box_info = self.proposals.get(image_id, [])

        regions = []
        bboxes = []
        for i in range(len(box_info)):
            region_path = os.path.join(region_dir, f'region_{i}.jpg')
            if not os.path.exists(region_path):
                continue
            image = Image.open(region_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            regions.append(image)
            bboxes.append(box_info[i]['box'])  # [x1, y1, x2, y2]

        return {
            'sentence': sentence,
            'gt_bbox': gt_bbox,
            'regions': regions,         # list of PIL image or transformed tensor
            'region_bboxes': bboxes,   # list of [x1, y1, x2, y2]
            'image_id': image_id,
            'file_name': sample['file_name']
        }
