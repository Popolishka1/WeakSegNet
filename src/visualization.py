import torch
import matplotlib.pyplot as plt

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
                axes[image_to_show, 1].set_title("Ground truth mak")
                axes[image_to_show, 2].imshow(pred_mask, cmap="gray")
                axes[image_to_show, 2].set_title("Predicted mask")

                for j in range(3):
                    axes[image_to_show, j].axis("off")

                image_to_show += 1
