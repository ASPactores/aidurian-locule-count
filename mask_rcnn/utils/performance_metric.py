from typing import Tuple

import numpy as np
from mrcnn.utils import compute_ap, compute_ap_range, get_iou
from mrcnn import model as modellib
import matplotlib.pyplot as plt


def compute_performance(
    image_ids, dataset, config, model
) -> Tuple[float, float, float, float]:
    ap_50_arr = []
    ap_75_arr = []
    map_arr = []

    iou_arr = []
    image_data = {}
    selected_image_ids = np.random.choice(image_ids, 2)  # Select 2 random image IDs

    for image_id in image_ids:
        # Load image
        image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id
        )

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        print(type(ap_50_arr))
        # Compute AP@0.50
        ap_50, _, _, _ = compute_ap(
            gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
            iou_threshold=0.50,
        )
        print(type(ap_50))
        print(ap_50)
        print(ap_50_arr)
        ap_50_arr.append(ap_50)

        # Compute AP@0.75
        ap_75, _, _, _ = compute_ap(
            gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
            iou_threshold=0.75,
        )
        ap_75_arr.append(ap_75)

        # Compute mAP
        map_val = compute_ap_range(
            gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
            verbose=0,
        )
        map_arr.append(map_val)

        # Compute IOU for each bounding box
        bounding_boxes = []

        for gt_box in gt_bbox:
            gt_y0, gt_x1, gt_y2, gt_x2 = gt_box
            best_iou = -1
            best_pred_box = None

            # Find the predicted box with the highest IoU for this ground truth box
            for pred_box in r["rois"]:
                pred_y0, pred_x1, pred_y2, pred_x2 = pred_box
                iou = get_iou(
                    [gt_x1, gt_y0, gt_x2, gt_y2], [pred_x1, pred_y0, pred_x2, pred_y2]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_pred_box = pred_box

            iou_arr.append(best_iou)
            bounding_boxes.append((image, gt_box, best_pred_box))

        # Store the bounding boxes for the current image_id if it's selected
        if image_id in selected_image_ids:
            image_data.setdefault(image_id, []).extend(bounding_boxes)

    # Display all images in a grid
    display_bounding_boxes(image_data)

    return np.mean(ap_50_arr), np.mean(ap_75_arr), np.mean(map_arr), np.mean(iou_arr)


"""
==========================
Visualizing Bounding Boxes
==========================
"""


def draw_bounding_boxes(ax, image, gt_box, best_pred_box, image_id):
    """
    Draw ground truth and predicted bounding boxes on the given axes.

    Args:
        ax: The axes to draw on.
        image: The image to display.
        gt_box: The ground truth bounding box in the format [y0, x1, y2, x2].
        best_pred_box: The best predicted bounding box in the format [y0, x1, y2, x2].
        image_id: The ID of the image.
    """
    ax.imshow(image)

    # Draw ground truth box
    gt_y0, gt_x1, gt_y2, gt_x2 = gt_box
    ax.add_patch(
        plt.Rectangle(
            (gt_x1, gt_y0),
            gt_x2 - gt_x1,
            gt_y2 - gt_y0,
            fill=False,
            edgecolor="red",
            lw=2,
            label="Ground Truth",
        )
    )

    # Draw predicted box if it exists
    if best_pred_box is not None:
        pred_y0, pred_x1, pred_y2, pred_x2 = best_pred_box
        ax.add_patch(
            plt.Rectangle(
                (pred_x1, pred_y0),
                pred_x2 - pred_x1,
                pred_y2 - pred_y0,
                fill=False,
                edgecolor="blue",
                lw=2,
                label="Prediction",
            )
        )

    ax.set_title(f"Image ID: {image_id}")


def display_bounding_boxes(image_data):
    """
    Display images with their ground truth and predicted bounding boxes in a grid layout.

    Args:
        image_data: A dictionary where keys are image IDs and values are lists of tuples
                    containing (image, gt_box, best_pred_box).
    """
    max_columns = 6  # Maximum number of columns
    rows = len(image_data)  # Number of unique image IDs
    grid_size = (rows, max_columns)  # Grid size (rows, columns)

    fig, axes = plt.subplots(*grid_size, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten the grid for easy iteration

    for idx, (image_id, bounding_boxes) in enumerate(image_data.items()):
        for col in range(max_columns):
            ax = axes[idx * max_columns + col]
            if col < len(bounding_boxes):
                image, gt_box, best_pred_box = bounding_boxes[col]
                draw_bounding_boxes(ax, image, gt_box, best_pred_box, image_id)
            else:
                ax.axis("off")  # Hide axes if no image available

    plt.tight_layout(pad=0.5, h_pad=0.5)  # Adjust layout
    plt.show()
