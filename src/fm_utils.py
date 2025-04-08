# TODO (Carl): generate weak mask from foundations models
import json
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from src.metrics import dice_score, iou_score, pixel_accuracy, precision_recall


def parse_args(experiment_name: str):
    if experiment_name == "SAM":
        parser = argparse.ArgumentParser(description="SAM segmentation with foundation models (weakly supervised use case)")
    else:
        raise ValueError("Experiment name not recognized. Consider switching to the non foundation models utils.")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to .json config file containing all experiment settings"
                        )
    args = parser.parse_args()
    # Load the configuration from the .json file
    with open(args.config, "r") as f:
        config = json.load(f)
    return config


# Assuming your metric functions (dice_score, iou_score, etc.) are defined elsewhere and imported.

def evaluate_and_visualise_sam(model, dataloader, threshold=0.5, device="cuda", n_samples=1):
    """
    For each image from the dataloader, runs the SAM model to get the prediction,
    then calculates and prints the metrics, and finally visualizes the input image,
    ground truth mask, and the SAM prediction.
    
    Args:
        model (callable): Your SAMWrapper (or equivalent) that processes one image (or batch).
        dataloader (DataLoader): Test DataLoader.
        threshold (float): Threshold for segmentation mask binarisation.
        device (str): Device to run inference on.
        n_samples (int): Number of individual images to process and visualize.
    """
    model.eval()  # Although SAM doesn't need eval mode, this keeps a consistent interface.
    sample_count = 0
    metrics = np.zeros(5)
    with torch.no_grad():
        for images, gt_masks, _ in dataloader:
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            batch_size = images.shape[0]
            for idx in range(batch_size):
                # Process a single image.
                single_image = images[idx].unsqueeze(0)  # Add batch dimension.
                pred_mask = model(single_image).squeeze(0)  # Remove the batch dimension.
                
                # Compute metrics for this individual image.
                dice = dice_score(pred_mask, gt_masks[idx], threshold)
                iou = iou_score(pred_mask, gt_masks[idx], threshold)
                accuracy = pixel_accuracy(pred_mask, gt_masks[idx], threshold)
                precision, recall = precision_recall(pred_mask, gt_masks[idx], threshold)
                
                print(f"Image {sample_count+1} Metrics:")
                print(f"  Dice: {dice:.4f}")
                print(f"  IoU: {iou:.4f}")
                print(f"  Pixel Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                
                # Add to array of metrics
                metrics[0] += dice
                metrics[1] += iou
                metrics[2] += accuracy
                metrics[3] += precision
                metrics[4] += recall
                
                # Convert tensors to NumPy arrays for visualization.
                image_np = images[idx].cpu().permute(1, 2, 0).numpy()
                gt_mask_np = gt_masks[idx].cpu().squeeze().numpy()
                # Apply threshold to the predicted mask.
                pred_mask_np = (pred_mask > threshold).float().cpu().numpy()
                
                # Create subplots for input image, ground truth, and prediction.
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(image_np)
                ax[0].set_title("Input Image")
                ax[0].axis("off")
                
                ax[1].imshow(gt_mask_np, cmap="gray")
                ax[1].set_title("Ground Truth Mask")
                ax[1].axis("off")
                
                ax[2].imshow(image_np)
                ax[2].imshow(pred_mask_np, alpha=0.5, cmap="gray")
                ax[2].set_title("SAM Prediction")
                ax[2].axis("off")
                
                plt.show()
                
                sample_count += 1
                # if sample_count >= n_samples:
                #     return
    return np.array(metrics) / sample_count