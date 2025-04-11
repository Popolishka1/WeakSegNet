import cv2
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import inverse_normalize
from src.cam_utils import  cam_to_binary_mask, get_cam_generator, apply_dense_crf


def create_overlay(img, gt_mask, pred_mask, alpha=0.5):
    """
    Create an overlay where: gt mask is blended in green and predicted mask is blended in red.
    Args:
        img: input in [0,1] format
        gt_mask: binary mask (0 or 1)
        pred_mask: binary mask (0 or 1)
        alpha: blending factor 
    Returns:
        overlay: float np array with blended colors
    """
    overlay = img.copy() # don't modify the original so make a copy

    green = np.array([0.0, 1.0, 0.0]) # for gt mask
    red = np.array([1.0, 0.0, 0.0]) # for predicted mask

    def blend(region, color, alpha):
        return region * (1 - alpha) + color * alpha

    overlay[gt_mask == 1] = blend(overlay[gt_mask == 1], green, alpha) # blend green where gt_mask == 1
    overlay[pred_mask == 1] = blend(overlay[pred_mask == 1], red, alpha) # blend red where pred_mask == 1

    return overlay


def visualise_predictions(config, dataloader, model, n_samples=5, threshold=0.5, device="cuda"):
    print(f"\n----Visualizing {n_samples} predicted masks from the trained model")
    model.eval()
    all_samples = [] # samples from the test loader
    with torch.no_grad():
        for images, gt_masks, info in dataloader:
            images = images.to(device)  
            gt_masks = gt_masks.to(device)
            batch_size = images.size(0)
            for i in range(batch_size):
                all_samples.append((images[i], gt_masks[i], info["name"][i]))
    
    # Shuffle and randomly select n_samples
    random.shuffle(all_samples)
    selected_samples = all_samples[:n_samples]
    
    fig, ax = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))
    
    if n_samples == 1:
        ax = np.expand_dims(ax, axis=0)

    for image_to_show, (img_tensor, gt_mask, img_name) in enumerate(selected_samples):
        # Inverse normalize image for display 
        img = inverse_normalize(img_tensor).cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # Get model prediction
        outputs = model(img_tensor.unsqueeze(0))
        predicted_mask = (outputs > threshold).float()
        pred_mask = predicted_mask[0].cpu().squeeze().numpy()
        gt_mask_np = gt_mask.cpu().squeeze().numpy()

        # Create an overlay image
        overlayed_img = create_overlay(img, gt_mask_np, pred_mask, alpha=0.5)

        # Plot: original image, gt mask and predicted mask
        ax[image_to_show, 0].imshow(img)
        ax[image_to_show, 0].set_title(f"Original image:\n{img_name}")
        
        ax[image_to_show, 1].imshow(gt_mask_np, cmap="gray")
        ax[image_to_show, 1].set_title("GT mask")
        
        ax[image_to_show, 2].imshow(pred_mask, cmap="gray")
        ax[image_to_show, 2].set_title("Predicted mask")
        
        ax[image_to_show, 3].imshow(overlayed_img)
        ax[image_to_show, 3].set_title("Overlay: \nGT (green) + Pred (red)")
        
        for j in range(4):
            ax[image_to_show, j].axis("off")
    
    save_path = config["segmentation_visualisation_save_path"]
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved predictions to {save_path}]")


def visualise_cams(config, dataloader, classifier, cam_type='CAM', cam_threshold=0.5, n_samples=5, device="cuda"):
    """Visualize CAM overlays for the first n_samples images, optionally with Dense CRF refinement."""
    print(f"\n----Visualizing CAMs for {n_samples} sample images")
    classifier.eval()

    use_dense_crf = config.get("use_dense_crf", False)
    if use_dense_crf:
        print("[Dense CRF visualization enabled - refining each sample's CAM]")

    target_id = config["target_id"]
    all_samples = []  # gather samples from the loader
    with torch.no_grad():
        for images, _, info in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            for i in range(batch_size):
                all_samples.append((images[i], info["name"][i], info[target_id][i]))
    
    # Shuffle and pick n_samples
    random.shuffle(all_samples)
    selected_samples = all_samples[:n_samples]

    # Create a figure with either 3 or 4 columns depending on CRF
    columns = 4 if use_dense_crf else 3
    fig, ax = plt.subplots(n_samples, columns, figsize=(5 * columns, 4 * n_samples))
    if n_samples == 1:
        ax = np.expand_dims(ax, axis=0)

    # Initialize CAM generator
    cam_generator, generate_cam_func, _ = get_cam_generator(classifier=classifier, cam_type=cam_type)
    
    # Visualize each sample
    for idx, (img_tensor, img_name, img_target) in enumerate(selected_samples):
        # Generate the raw CAM
        if cam_type == 'CAM':
            with torch.no_grad():
                cam, pred_class = generate_cam_func(cam_generator, img_tensor)
        else:
            with torch.enable_grad():
                cam, pred_class = generate_cam_func(cam_generator, img_tensor)

        # Inverse normalize
        img = inverse_normalize(img_tensor).cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # Create a heatmap overlay from the raw CAM
        cam_np = cam.detach().cpu().numpy() if cam.requires_grad else cam.cpu().numpy()
        heatmap = np.uint8(255 * cam_np)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlayed_img = cv2.addWeighted(np.uint8(255 * img), 0.5, heatmap, 0.5, 0)
        
        # Create raw pseudo mask
        pseudo_mask = cam_to_binary_mask(cam, cam_threshold).squeeze().cpu().numpy()
        
        # Plot columns: 1) Original 2) CAM overlay 3) Raw pseudo mask
        ax[idx, 0].imshow(img)
        ax[idx, 0].set_title(f"Original: {img_name}\n(GT class {img_target + 1})")
        ax[idx, 1].imshow(overlayed_img)
        ax[idx, 1].set_title(f"CAM overlay\n(Pred class {pred_class + 1})")
        ax[idx, 2].imshow(pseudo_mask, cmap="gray")
        ax[idx, 2].set_title(f"Pseudo mask @ {cam_threshold}")

        # Optionally refine via Dense CRF and plot
        if use_dense_crf:
            refined_cam = apply_dense_crf(img_tensor, cam)
            refined_mask = cam_to_binary_mask(refined_cam, cam_threshold).squeeze().cpu().numpy()
            ax[idx, 3].imshow(refined_mask, cmap="gray")
            ax[idx, 3].set_title(f"CRF-refined mask @ {cam_threshold}")

        for j in range(columns):
            ax[idx, j].axis("off")

    # Ensure directory exists
    save_path = config.get("cam_visualisation_save_path", "results/cam/cam_examples.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved CAM visualisations to {save_path}]")
