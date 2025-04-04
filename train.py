# train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.train_dataset import RefExpTrainDataset
from models.clip_matcher import CLIPMatcher
import clip
from livelossplot import PlotLosses

# Collate function to stack variable-length region images
def collate_fn(batch):
    region_images_batch = []
    texts = []
    targets = []
    for sample in batch:
        imgs = sample['region_images']
        if len(imgs) == 0:
            continue
        tensor_imgs = torch.stack(imgs)  # (R_i, 3, H, W)
        region_images_batch.append(tensor_imgs)
        texts.append(sample['text'])
        targets.append(sample['target_index'])
    return region_images_batch, texts, torch.tensor(targets)

# Paths
ref_json = "data/refcoco_mini.json"
proposals_json = "outputs/proposals.json"
regions_dir = "outputs/regions"

# Transforms
_, clip_transform = clip.load("ViT-B/32")

# Dataset and DataLoader
dataset = RefExpTrainDataset(ref_json, proposals_json, regions_dir, transform=clip_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPMatcher(freeze_clip=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
liveloss = PlotLosses()

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    logs = {}

    for region_imgs, texts, targets in dataloader:
        region_imgs = [imgs.to(device) for imgs in region_imgs]
        targets = targets.to(device)

        scores_list = model(region_imgs, texts)
        losses = []
        for scores, target in zip(scores_list, targets):
            scores = scores.unsqueeze(0)  # (1, N)
            target = target.unsqueeze(0)  # (1)
            loss = criterion(scores, target)
            losses.append(loss)
            pred = scores.argmax(dim=1).item()
            if pred == target.item():
                correct += 1
            total += 1

        batch_loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    logs['loss'] = avg_loss
    logs['accuracy'] = accuracy
    liveloss.update(logs)
    liveloss.send()

    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
