import torch


def dice_score(output, true_mask, threshold=0.5, eps=1e-7):
    """
    One of the possible metric that we can use: dice score between predicted masks and ground truth.
    """
    predicted_mask = (output > threshold).float()
    true_mask = true_mask.float()

    intersection = (predicted_mask * true_mask).sum(dim=(1, 2, 3))
    union = predicted_mask.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def pixel_accuracy(output, true_mask, threshold=0.5):
    predicted_mask = (output > threshold).float()
    true_mask = true_mask.float()

    correct = (predicted_mask == true_mask).float()

    accuracy = correct.sum(dim=(1, 2, 3)) / correct[0].numel()
    return accuracy.mean().item()


# TODO: add other metrics: IoU (good for overlap), the good boy precision & recall maybe as well


def evaluate_model(model, test_loader, device="cpu"):
    print("\n----Evaluating the model on the test set")
    model.eval()
    total_dice = 0.0
    total_accuracy = 0.0
    n_batches = len(test_loader)

    with torch.no_grad():
        for images, true_masks, _ in test_loader:
            images = images.to(device)
            true_masks = true_masks.to(device)

            outputs = model(images)

            dice = dice_score(outputs, true_masks)
            accuracy = pixel_accuracy(outputs, true_masks)

            total_dice += dice
            total_accuracy += accuracy

    avg_dice = total_dice / n_batches
    avg_accuracy = total_accuracy / n_batches
    print(f"Average test set dice score: {avg_dice:.4f}") # TODO: should we consider a global evaluation? so total performance instead of per-image
    print(f"Average test set pixel accuracy: {avg_accuracy:.4f}")
    return avg_dice, avg_accuracy