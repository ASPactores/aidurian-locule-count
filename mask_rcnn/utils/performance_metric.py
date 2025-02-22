import numpy as np
import matplotlib.pyplot as plt
from mrcnn import utils
from mrcnn.utils import get_iou
from mrcnn import model as modellib
from mrcnn.visualize import plot_precision_recall


def compute_performance_metrics(
    image_ids, test_model, dataset_test, test_config, iou_thresholds=[0.50, 0.75]
):
    AP_50 = []
    AP_75 = []
    mAPs = []
    IOUs = []
    image_data = {}

    all_precisions = []
    all_recalls = []

    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset_test, test_config, image_id
        )
        results = test_model.detect([image], verbose=0)
        r = results[0]

        ap_50, precisions_50, recalls_50, _ = utils.compute_ap(
            gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
            iou_threshold=0.50,
        )
        ap_75, _, _, _ = utils.compute_ap(
            gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
            iou_threshold=0.75,
        )

        AP_50.append(ap_50)
        AP_75.append(ap_75)

        # Compute mean AP across thresholds
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

        # Store precision-recall pairs for later aggregation
        all_precisions.append(precisions_50)
        all_recalls.append(recalls_50)

    IOUs = np.mean(IOUs) if IOUs else 0

    # Normalize and fix the precision-recall curve
    recall_bins = np.linspace(0, 1, 101)  # Define standard recall bins
    precision_interpolated = []

    for precisions, recalls in zip(all_precisions, all_recalls):
        # Sort recall and precision pairs
        sorted_indices = np.argsort(recalls)
        recalls = np.array(recalls)[sorted_indices]
        precisions = np.array(precisions)[sorted_indices]

        # Interpolate precision at standard recall bins
        precision_interp = np.interp(recall_bins, recalls, precisions, left=0, right=0)
        precision_interpolated.append(precision_interp)

    # Average interpolated precision across all images
    mean_precision = np.mean(precision_interpolated, axis=0)

    # Plot Precision-Recall Curve
    plt.figure(figsize=(6, 6))
    plt.plot(
        recall_bins, mean_precision, label=f"Precision-Recall Curve", color="black"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: ")

    # Move legend outside the plot
    plt.legend(
        [f"Locules: {np.mean(AP_50):.3f}"], loc="upper right", bbox_to_anchor=(1.3, 1)
    )

    plt.grid(False)

    # Remove top and right border
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()

    return {
        "AP_50": np.mean(AP_50),
        "AP_75": np.mean(AP_75),
        "mAP": np.mean(mAPs),
        "IOUs": IOUs,
    }


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
