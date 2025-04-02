import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from cam_utils import CAMGenerator, generate_cam


def visualize_predictions(model, test_loader, n_samples=5, threshold=0.5, device="cpu", save_path="predictions.png"):
    print(f"\n----Visualizing {n_samples} predicted masks from the trained model")
    model.eval()
    image_to_show = 0
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    
    with torch.no_grad():
        for images, true_masks, _ in test_loader:
            images = images.to(device)
            true_masks = true_masks.to(device)

            outputs = model(images)
            predicted_mask = (outputs > threshold).float()

            for i in range(images.size(0)):
                if image_to_show >= n_samples:
                    fig.tight_layout()
                    plt.savefig(save_path)
                    plt.close()
                    print(f"[Saved predictions to {save_path}]")
                    return

                img = images[i].cpu().permute(1, 2, 0).numpy()
                gt_mask = true_masks[i].cpu().squeeze().numpy()
                pred_mask = predicted_mask[i].cpu().squeeze().numpy()

                axes[image_to_show, 0].imshow(img)
                axes[image_to_show, 0].set_title("Original image") # TODO: include the name of the image maybe
                axes[image_to_show, 1].imshow(gt_mask, cmap="gray")
                axes[image_to_show, 1].set_title("Ground truth mask")
                axes[image_to_show, 2].imshow(pred_mask, cmap="gray")
                axes[image_to_show, 2].set_title("Predicted mask")

                for j in range(3):
                    axes[image_to_show, j].axis("off")

                image_to_show += 1


def visualize_cams(classifier, test_loader, n_samples=5, cam_threshold=0.5, save_path="cam_visualization.png", device="cpu"):
    """Visualize CAM overlays for the first n_samples images."""

    print(f"\n----Visualizing CAMs for {n_samples} sample images")
    classifier.eval()
    image_to_show = 0
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    # Initialize CAM generator
    cam_generator = CAMGenerator(classifier)
    
    with torch.no_grad():
        for images, _, info in test_loader:
            images = images.to(device)
            
            for i in range(images.size(0)):
                if image_to_show >= n_samples:
                    break
                
                # Get image and generate CAM
                img_tensor = images[i]
                cam, pred_class = generate_cam(cam_generator, img_tensor)
                
                # Convert tensors to numpy
                img = img_tensor.cpu().permute(1, 2, 0).numpy()
                cam_np = cam.cpu().numpy()
                
                # Create overlay
                heatmap = np.uint8(255 * cam_np)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                overlayed_img = cv2.addWeighted(
                    cv2.cvtColor(np.uint8(255 * img), 0.5, 
                    heatmap, 0.5, 0
                ))
                
                # Create binary mask
                binary_mask = (cam_np > cam_threshold).astype(np.float32)
                
                # Plotting
                axes[image_to_show, 0].imshow(img)
                axes[image_to_show, 0].set_title("Original image")
                
                axes[image_to_show, 1].imshow(overlayed_img)
                axes[image_to_show, 1].set_title(f"CAM overlay (Class {pred_class})")
                
                axes[image_to_show, 2].imshow(binary_mask, cmap="gray")
                axes[image_to_show, 2].set_title(f"Binary mask @ {cam_threshold}")
                
                for j in range(3):
                    axes[image_to_show, j].axis("off")
                
                image_to_show += 1

            if image_to_show >= n_samples:
                break

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved CAM visualizations to {save_path}]")