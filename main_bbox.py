import torch
from torch.utils.data import DataLoader

from src.bbox_utils import generate_pseudo_masks
from src.models import select_segmentation_model
from src.dataset import load_data_wrapper, PseudoMaskDataset
from src.fit import train_classifier, train_segmentation_model
from src.utils import load_device, clear_cuda_cache, parse_args
from src.visualisation import visualise_predictions, visualise_cams
from src.metrics import evaluate_model, evaluate_classifier, save_metrics_to_csv


def main():
    # Load configuration from .json config file (BBOX experiment)
    config = parse_args(expriment_name="BBOX")

    train_loader, val_loader, test_loader = load_data_wrapper(config=config)

    
    total_dice = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n_batches = len(train_loader)
    batch_size = batch_size_test
    print(n_batches)
    output_dir = "saved_pseudo_masks"
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx, (image_batch, mask_batch, info_batch) in enumerate(train_loader, start=1):
        print(f"Batch number: {batch_idx}")
        cams = generate_cams(image_batch)
        bboxs = generate_bboxs(cams, image_batch)
        pseudo_masks = generate_pseudo_masks(bboxs, image_batch, variant="GrabCut")
        dice, accuracy_score, iou, precision, recall = evaluate(pseudo_masks, mask_batch, batch_size)
        print(f'Dice score is {dice} || Pixel score is {accuracy_score} || IOU score is {iou} || Precision score is {precision} || Recall score is {recall}')
        total_dice += dice
        total_accuracy += accuracy_score
        total_iou += iou
        total_precision += precision
        total_recall += recall
        visualise_results(image_batch[0], bboxs[0], mask_batch[0], pseudo_masks[0], variant="GrabCut")
        save_pseudo_masks(pseudo_masks, batch_idx, output_dir)
    
    print(f'Average dice score per image is {total_dice / n_batches}')
    print(f'Average accuracy score per image is {total_accuracy / n_batches}')
    print(f'Average IOU score per image is {total_iou / n_batches}')
    print(f'Average precision score per image is {total_precision / n_batches}')
    print(f'Average recall score per image is {total_recall / n_batches}')


if __name__ == "__main__":
    main()
