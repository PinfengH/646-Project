# 646-Project

# Referring Expression Comprehension with CLIP

This project implements a pipeline for Referring Expression Comprehension (REC) using OpenAI's CLIP model. The objective is to localize the object described by a natural language expression within an image, based on region proposals and multimodal matching.

---

## Project Structure

<details>
<summary>Click to expand full folder structure</summary>

```
646_project/
├── data/
│   ├── annotations/                  # From MS COCO annotations zip
│   ├── images/                       # COCO images: train2014/, val2014/
│   ├── refcoco/                      # Raw RefCOCO dataset (from UNC website)
│   └── refcoco_mini/                 # Processed JSON from our code (not downloadable)
│
├── outputs/                          # Saved proposals, predictions, and visualizations
│   ├── proposals.json                # Region proposals from Faster R-CNN
│   ├── regions/                      # Region crops per image
│   ├── predictions_clip.json         # CLIP matching results
│   ├── vis_clip/                     # Visualization output
│
├── models/
│   └── clip_matcher.py               # CLIP-based model wrapper
│
├── utils/
│   ├── datasets.py                   # Dataset for prediction
│   ├── train_dataset.py              # Dataset for supervised training
│   └── vis.py                        # Visualization helper
│
├── tools/
│   └── eval_clip.py                  # Evaluation script
│
├── stage1_proposal.py                # Generate region proposals and save crops
├── predict_clip.py                   # Run CLIP-based matching and save results
├── vis_clip.py                       # Visualize predicted vs GT boxes
├── train.py                          # Train the CLIP matching model
└── README.md                         # This file
```

</details>

---

## How to Prepare the Data

This project relies on data from MS COCO and RefCOCO. Since these datasets are large and cannot be hosted directly on GitHub, please follow the instructions below to download and prepare the necessary files.

### 1. MS COCO Dataset
- Download the **2014 Train/Val images** and **annotations** from the official site:
  - Images: https://cocodataset.org/#download
    - `train2014.zip`
    - `val2014.zip`
  - Annotations:
    - `annotations_trainval2014.zip`
- Unzip and place into `data/`:

```
data/
├── images/
│   ├── train2014/
│   └── val2014/
├── annotations/
│   └── instances_train2014.json, captions_val2014.json, ...
```

### 2. RefCOCO Dataset (from UNC)
- Download from: [https://github.com/lichengunc/refer](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)
- Clone the repo and follow their instructions to download the following files:
  - `refs(unc).p`, `instances.json`, `images/`
- Place them in:
```
data/
└── refcoco/
```

>   You may need to manually request access or process `.p` files using their loader script.

### 3. Generate `refcoco_mini` (processed JSON)
After preparing the data, run `convert_refcoco_to_json.py` or related scripts in this repo. It will automatically process the RefCOCO `.p` files and COCO images to produce a JSON subset like:
```
data/refcoco_mini/refcoco_mini.json
```
This file contains all referring expressions and bounding boxes in a flattened JSON format ready for training and evaluation.

---

## Pipeline Overview

### 1. Region Proposal
Run `stage1_proposal.py` to generate region proposals using a fine-tuned Faster R-CNN.

### 2. CLIP-Based Region Scoring
Use `predict_clip.py` to compute cosine similarity between region crops and referring expressions using frozen CLIP encoders.

### 3. Evaluation
Run `tools/eval_clip.py` to compute accuracy and mean IoU.

### 4. Visualization
Use `vis_clip.py` to visualize predicted vs. ground truth bounding boxes.

> *Note:* We also implemented `train.py` for CLIP-based matching training, but results were suboptimal compared to using frozen CLIP encoders

---

## Results

- Accuracy improved from **41.2%** (baseline) to **54.7%**
- Mean IoU improved from **0.407** to **0.539**
- Best results were achieved using frozen CLIP + fine-tuned Faster R-CNN proposals

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenAI CLIP:
  ```bash
  pip install git+https://github.com/openai/CLIP.git
  ```
- livelossplot:
  ```bash
  pip install livelossplot
  ```

---

## Author
Pinfeng Huang 

Graduate Student @ Rice University  

ph60@rice.edu
