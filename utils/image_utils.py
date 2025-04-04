# utils/image_utils.py

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def crop_regions(image, boxes):
    """
    Crop regions from an image based on bounding boxes.

    Args:
        image (PIL.Image): The original image.
        boxes (Tensor or list): List of bounding boxes [x1, y1, x2, y2].

    Returns:
        crops (list): List of cropped region images (PIL.Image).
    """
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        region = image.crop((x1, y1, x2, y2))
        crops.append(region)
    return crops


def draw_boxes(image, boxes, scores=None):
    """
    Visualize bounding boxes on an image.

    Args:
        image (PIL.Image): The original image.
        boxes (Tensor or list): Bounding boxes [x1, y1, x2, y2].
        scores (list, optional): Confidence scores to display.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if scores is not None:
            ax.text(x1, y1, f"{scores[i]:.2f}", color='white',
                    fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
