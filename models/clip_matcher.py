# models/clip_matcher.py

import torch
import torch.nn as nn
import clip

class CLIPMatcher(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", freeze_clip=True):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.image_encoder = self.clip_model.encode_image
        self.text_encoder = self.clip_model.encode_text

        #if freeze_clip:
        #    for param in self.clip_model.parameters():
        #        param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones([]) * 1.0)

    def forward(self, region_images, text):
        """
        region_images: list of (B, 3, H, W) tensors with different region counts
        text: list of sentences (length B)
        """
        device = next(self.parameters()).device
        text_tokens = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_features = self.text_encoder(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # (B, D)

        batch_scores = []
        for i in range(len(region_images)):
            regions = region_images[i].to(device)  # (R_i, 3, H, W)
            with torch.no_grad():
                region_feats = self.image_encoder(regions).float()
            region_feats = region_feats / region_feats.norm(dim=-1, keepdim=True)  # (R_i, D)

            # text_features[i]: (1, D) vs region_feats: (R_i, D)
            sim = region_feats @ text_features[i].unsqueeze(0).T  # (R_i, 1)
            scores = sim.squeeze(1) * self.logit_scale.exp()  # (R_i)
            batch_scores.append(scores)

        return batch_scores  # list of (R_i,) scores for each sample
