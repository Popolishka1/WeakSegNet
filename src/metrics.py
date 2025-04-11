import os
import csv
import torch
from datetime import datetime


def dice_score(output, gt_mask, threshold=0.5, eps=1e-7):
    """
    One of the possible metric that we can use: dice score between predicted masks and ground truth.
    """
    predicted_mask = (output > threshold).float()
    gt_mask = gt_mask.float()

    intersection = (predicted_mask * gt_mask).sum()
    union = predicted_mask.sum() + gt_mask.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return dice.item()


def iou_score(output, gt_mask, threshold=0.5, eps=1e-7):
    predicted_mask = (output > threshold).float()
    gt_mask = gt_mask.float()

    intersection = (predicted_mask * gt_mask).sum()
    union = predicted_mask.sum() + gt_mask.sum() - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.item()


def pixel_accuracy(output, gt_mask, threshold=0.5):
    predicted_mask = (output > threshold).float()
    gt_mask = gt_mask.float()

    correct = (predicted_mask == gt_mask).float()

    total = torch.numel(gt_mask)
    accuracy = correct.sum() / total
    return accuracy.item()


def precision_recall(output, gt_mask, threshold=0.5, eps=1e-7):
    predicted_mask = (output > threshold).float()
    gt_mask = gt_mask.float()

    tp = (predicted_mask * gt_mask).sum()
    predicted_positive = predicted_mask.sum()
    actual_positive = gt_mask.sum()

    precision = (tp + eps) / (predicted_positive + eps)
    recall = (tp + eps) / (actual_positive + eps)

    return precision.item(), recall.item()


def evaluate_model(model, test_loader, threshold=0.5, device="cuda"):
    """
    Evaluate segmentation model performance on test set.
    Gloval evaluation of the model performance (metrics are not averaged per batch vut global).
    """
    print("\n----Evaluating the segmentation model on the test set...")
    model.eval()
    all_outputs = []
    all_gt_masks = []

    with torch.no_grad():
        for images, gt_masks, _ in test_loader:
            images = images.to(device)
            gt_masks = gt_masks.to(device)
                        
            outputs = model(images)

            all_outputs.append(outputs.cpu())
            all_gt_masks.append(gt_masks.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_gt_masks = torch.cat(all_gt_masks, dim=0)

    dice = dice_score(all_outputs, all_gt_masks, threshold)
    iou = iou_score(all_outputs, all_gt_masks, threshold)
    accuracy = pixel_accuracy(all_outputs, all_gt_masks, threshold)
    precision, recall = precision_recall(all_outputs, all_gt_masks, threshold)

    print("Performance metrics on the test set:")
    print(f"Dice score: {dice:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Pixel accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    metrics ={"dice": dice,
              "iou": iou,
              "accuracy": accuracy,
              "precision": precision,
              "recall": recall
              }
    return metrics


def save_metrics_to_csv(metrics, save_path="metrics.csv"):
    """Simple function to save metrics to a CSV file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    write_header = not os.path.exists(save_path)

    with open(save_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def evaluate_classifier(classifier, test_loader, config, device="cuda"):
    """Evaluate classifier performance on test set."""
    print("\n----Evaluating classifier on the test set...")
    classifier.eval()
    correct = 0
    total = 0

    target_id = config["target_id"]
    with torch.no_grad():
        for images, _, info in test_loader:
            images = images.to(device)
            labels = info[target_id].to(device).long()
            outputs = classifier(images)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total

    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy