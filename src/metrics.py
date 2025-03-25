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


def evaluate_model(model, testloader, device="cpu"):
    model.eval()
    total_dice = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks in testloader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            dice = dice_score(preds, masks) # TODO: add other metrics
            total_dice += dice
            n_batches += 1

    avg_dice = total_dice / n_batches
    print(f"Test set dice score: {avg_dice:.4f}")
    return avg_dice