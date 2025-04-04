# tools/filter_refcoco.py

import os
import json

# Load full refcoco_mini.json
with open("data/refcoco_mini.json", "r") as f:
    all_samples = json.load(f)

# Load proposals.json to get processed image_ids
with open("outputs/proposals.json", "r") as f:
    proposals = json.load(f)

valid_ids = set(proposals.keys())
print(f"Found {len(valid_ids)} image_ids with region proposals.")

# Filter only samples whose image_id is in valid_ids
filtered = []
for item in all_samples:
    file_name = item["file_name"]
    image_id = file_name.split("_")[-1].split(".")[0]  # e.g., 000000123456
    if image_id in valid_ids:
        filtered.append(item)

# Save filtered subset
save_path = "data/refcoco_filtered.json"
with open(save_path, "w") as f:
    json.dump(filtered, f, indent=2)

print(f"Filtered samples: {len(filtered)} saved to {save_path}")
