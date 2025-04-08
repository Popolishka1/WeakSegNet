import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import inverse_normalize
from src.cam_utils import  cam_to_binary_mask, get_cam_generator


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
    """Visualize CAM overlays for the first n_samples images."""
    print(f"\n----Visualizing CAMs for {n_samples} sample images")
    classifier.eval()

    target_id = config["target_id"]
    all_samples = [] # samples from the test loader
    with torch.no_grad():
        for images, _, info in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            for i in range(batch_size):
                # Save a tuple of (image tensor, image name)
                # Be careful: target classes are 0-indexed for PyTorch
                # But gt class are 1-indexed in the dataset 
                all_samples.append((images[i], info["name"][i], info[target_id][i]))
    
    # Shuffle and randomly select n_samples
    random.shuffle(all_samples)
    selected_samples = all_samples[:n_samples]

    fig, ax = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    if n_samples == 1:
        ax = np.expand_dims(ax, axis=0)

    cam_generator, generate_cam_func, cam_type = get_cam_generator(classifier=classifier, cam_type=cam_type)
    
    with torch.no_grad():
        for idx, (img_tensor, img_name, img_target) in enumerate(selected_samples):

            if cam_type == 'CAM':
                with torch.no_grad():
                    cam, pred_class = generate_cam_func(cam_generator, img_tensor)
            else:
                with torch.enable_grad():
                    cam, pred_class = generate_cam_func(cam_generator, img_tensor)

            # Inverse normalize image for display 
            img = inverse_normalize(img_tensor).cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            cam_np = cam.cpu().numpy()

            # Create a heatmap from CAM
            heatmap = np.uint8(255 * cam_np)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlayed_img = cv2.addWeighted(np.uint8(255 * img), 0.5, heatmap, 0.5, 0) # blend heatmap and image 

            # Create pseudo mask from CAM
            pseudo_mask = cam_to_binary_mask(cam=cam, cam_threshold=cam_threshold)

            # Plot: original image + CAM overlay + pseudo mask
            ax[idx, 0].imshow(img)
            # Be careful: 0-indexed for PyTorch, but target classes are 1-indexed in the dataset
            ax[idx, 0].set_title("Original image: " + img_name + f" (GT class {img_target + 1})") # +1 to match the dataset class index
            
            ax[idx, 1].imshow(overlayed_img)
            # Be careful: 0-indexed for PyTorch, but target classes are 1-indexed in the dataset
            ax[idx, 1].set_title(f"CAM overlay (Class pred {pred_class + 1})") # +1 to match the dataset class index
            
            ax[idx, 2].imshow(pseudo_mask.squeeze().cpu().numpy(), cmap="gray")
            ax[idx, 2].set_title(f"Pseudo mask from CAM @ {cam_threshold}")
        
            for j in range(3):
                ax[idx, j].axis("off")
                
    save_path = config["cam_visualisation_save_path"]
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved CAM visualisations to {save_path}]")