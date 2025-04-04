# utils/vis.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def draw_prediction(image_path, gt_bbox, pred_bbox, save_path=None, title=None):
    """
    Draw ground truth and predicted bounding boxes on the original image.

    Args:
        image_path (str): Path to the original COCO image.
        gt_bbox (list): Ground truth bbox in [x, y, w, h] format.
        pred_bbox (list): Predicted bbox in [x1, y1, x2, y2] format.
        save_path (str): Optional path to save the image.
        title (str): Optional title for the plot.
    """
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)

    # Ground Truth box: green
    x, y, w, h = gt_bbox
    gt_rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none', label='GT')
    ax.add_patch(gt_rect)
    ax.text(x, y - 5, 'GT', color='green', fontsize=10)

    # Predicted box: red
    x1, y1, x2, y2 = pred_bbox
    pred_rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none', label='Pred')
    ax.add_patch(pred_rect)
    ax.text(x1, y1 - 5, 'Pred', color='red', fontsize=10)

    ax.axis('off')
    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()
