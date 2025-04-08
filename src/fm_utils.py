# TODO (Carl): generate weak mask from foundations models
import json
import torch
import argparse
import matplotlib.pyplot as plt


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


def visualise_predictions_sam(config, dataloader, model, n_samples=5, threshold=0.5, device="cuda"):
    """
    Visualizes segmentation predictions from SAM.
    
    Args:
        config (dict): Configuration dictionary.
        dataloader (DataLoader): DataLoader for the test set.
        model (callable): A SAMWrapper (or equivalent) that, given a batch of images,
                          returns a stacked tensor of predicted masks.
        n_samples (int): Number of samples to visualize.
        threshold (float): Threshold applied to the predicted masks.
        device (str): Device used for predictions.
    """
    model.eval()  # though SAM doesn't require 'eval' mode, keep this for consistency.
    count = 0
    for images, gt_masks, info in dataloader:
        images = images.to(device)
        gt_masks = gt_masks.to(device)
        
        # Pass the batch through your SAM model wrapper.
        outputs = model(images)  # outputs is assumed to be a tensor of shape (B, H, W)
        
        for idx in range(outputs.shape[0]):
            # Convert image to NumPy for plotting; assume images are normalized 0-1.
            image_np = images[idx].cpu().permute(1, 2, 0).numpy()
            # Optionally, if you normalized your images, inverse normalize here.
            gt_mask_np = gt_masks[idx].cpu().squeeze().numpy()
            
            # Process the predicted mask: threshold and convert to numpy.
            pred_mask = outputs[idx]
            pred_mask = (pred_mask > threshold).float().cpu().numpy()
            
            # Create a figure with three subplots: input, ground truth, prediction overlay.
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            ax[0].imshow(image_np)
            ax[0].set_title("Input Image")
            ax[0].axis("off")
            
            ax[1].imshow(gt_mask_np, cmap="gray")
            ax[1].set_title("Ground Truth Mask")
            ax[1].axis("off")
            
            ax[2].imshow(image_np)
            ax[2].imshow(pred_mask, alpha=0.5, cmap="jet")
            ax[2].set_title("SAM Prediction")
            ax[2].axis("off")
            
            plt.show()
            
            count += 1
            if count >= n_samples:
                return