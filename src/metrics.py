import torch

def dice_score(preds, targets, threshold=0.5, eps=1e-7):
    """
    One of the possible metric that we can use: dice score between predicted masks and ground truth.
    """
    preds = (preds > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def pixel_accuracy(predicted_mask, target_mask, threshold=0.5):
    predicted_mask = (predicted_mask > threshold).float()
    correct = (predicted_mask == target_mask).float()
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
        for images, masks, _ in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            dice = dice_score(preds, masks)
            accuracy = pixel_accuracy(preds, masks)

            total_dice += dice
            total_accuracy += accuracy

    avg_dice = total_dice / n_batches
    avg_accuracy = total_accuracy / n_batches
    print(f"Average test set dice score: {avg_dice:.4f}") # TODO: should we consider a global evaluation? so total performance instead of per-image
    print(f"Average test set pixel accuracy: {avg_accuracy:.4f}")
    return avg_dice, avg_accuracy