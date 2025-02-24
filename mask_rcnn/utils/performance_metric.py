import numpy as np
import matplotlib.pyplot as plt
from mrcnn import utils
from mrcnn.utils import get_iou
from mrcnn import model as modellib


def compute_performance_metrics(
    image_ids, test_model, dataset_test, test_config, iou_thresholds=[0.50, 0.75]
):
    """
    Computes performance metrics (AP at IoU=0.50, AP at IoU=0.75, mean AP, and IoU)
    for the given image IDs.

    Args:
        image_ids: List of image IDs to evaluate.
        iou_thresholds: List of IoU thresholds for AP calculation (e.g., [0.50, 0.75]).

    Returns:
        A dictionary containing:
            - AP_50: List of APs for IoU=0.50
            - AP_75: List of APs for IoU=0.75
            - mAP: List of mean APs (average across thresholds)
            - IOUs: List of IoUs for each ground truth bounding box
            - image_data: Data needed for visualization, structured as:
                          {image_id: [(image, gt_box, best_pred_box), ...]}
    """
    AP_50 = []
    AP_75 = []
    mAPs = []
    IOUs = []
    image_data = {}

    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset_test, test_config, image_id
        )

        # Run object detection
        results = test_model.detect([image], verbose=0)
        r = results[0]

        # Compute AP for IoU thresholds 0.50 and 0.75
        AP_50.append(
            utils.compute_ap(
                gt_bbox,
                gt_class_id,
                gt_mask,
                r["rois"],
                r["class_ids"],
                r["scores"],
                r["masks"],
                iou_threshold=0.50,
            )[0]
        )
        AP_75.append(
            utils.compute_ap(
                gt_bbox,
                gt_class_id,
                gt_mask,
                r["rois"],
                r["class_ids"],
                r["scores"],
                r["masks"],
                iou_threshold=0.75,
            )[0]
        )

        # Compute mean AP across thresholds (using utils.compute_ap_range if available)
        mAPs.append(
            utils.compute_ap_range(
                gt_bbox,
                gt_class_id,
                gt_mask,
                r["rois"],
                r["class_ids"],
                r["scores"],
                r["masks"],
                verbose=0,
            )
        )

        # Prepare bounding boxes for IoU computation and visualization
        bounding_boxes = []
        for gt_box in gt_bbox:
            best_iou = -1
            best_pred_box = None
            for pred_box in r["rois"]:
                iou = get_iou(
                    [gt_box[1], gt_box[0], gt_box[3], gt_box[2]],
                    [pred_box[1], pred_box[0], pred_box[3], pred_box[2]],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_pred_box = pred_box
            IOUs.append(best_iou)
            bounding_boxes.append((image, gt_box, best_pred_box))

        # Store bounding boxes for visualization
        image_data[image_id] = bounding_boxes

    # Collect results in a dictionary
    metrics = {
        "AP_50": np.mean(AP_50),
        "AP_75": np.mean(AP_75),
        "mAP": np.mean(mAPs),
        "IOUs": np.mean(IOUs),
        "image_data": image_data,
    }
    return metrics


"""
================
Visualize Results
================
"""


def display_bounding_boxes(image_data, maximum_images=3):
    """
    Display images with their ground truth and predicted bounding boxes in a grid layout.

    Args:
        image_data: A dictionary where keys are image IDs and values are lists of tuples
                    containing (image, gt_box, best_pred_box).
    """
    max_columns = 6  # Maximum number of columns
    selected_image_ids = np.random.choice(
        list(image_data.keys()), maximum_images, replace=False
    )

    # Create a new dictionary with the selected items
    selected_images = {
        image_id: image_data[image_id] for image_id in selected_image_ids
    }
    image_data = selected_images

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
