import torch
from torch.utils.data import DataLoader
import os

from src.dataset import load_data_wrapper, PseudoMaskDataset
from src.utils import load_device, clear_cuda_cache, parse_args
from src.bbox_utils import generate_pseudo_mask_dataset_with_bbox, load_pseudo_mask
from src.models import select_segmentation_model
from src.fit import train_segmentation_model
from src.metrics import evaluate_model, save_metrics_to_csv


def main():

    config = parse_args(expriment_name="CAM")
    
    # Set device and clear CUDA cache
    DEVICE = load_device()
    clear_cuda_cache()

    ##################################################
    # 1. Load dataset (train, val, and test loaders) #
    ##################################################

    train_loader, val_loader, test_loader = load_data_wrapper(config=config)

    ############################################
    # 2. Load or generate pseudo-masks dataset #
    ############################################
    
    if config["dataset_output_dir"]:
        generate_pseudo_mask_dataset_with_bbox(train_loader, config=config)
    else:
        pseudo_masks = load_pseudo_mask(config["dataset_dir"])  

    ####################################
    # 3. Initialise segmentation model #
    ####################################
    segmentation_model = select_segmentation_model(config["segmentation_model"])
    if segmentation_model is None:
            pass
    segmentation_model.to(DEVICE)
    

    ###############################################################
    # 4. Train segmentation model using the Bounding Boxes Dataset#
    ###############################################################

    train_dataset_pseudo = PseudoMaskDataset(original_dataset=train_loader.dataset,
                                                 pseudo_masks=pseudo_masks
                                                 )
    train_loader_pseudo = DataLoader(train_dataset_pseudo, batch_size=config["train_batch_size"], shuffle=True, num_workers=0)
    
    # 4.2 Train or load baseline segmentation model
    if config["train_segmentation"]:
        print("\n----Training segmentation model...")
        segmentation_model = train_segmentation_model(model=segmentation_model,
                                                        train_loader=train_loader_pseudo,
                                                        val_loader=val_loader,
                                                        config=config,
                                                        device=DEVICE
                                                        )
        torch.save(segmentation_model.state_dict(), config["weakseg_model_save_path"])
        clear_cuda_cache()
    else:
        print("\n----Loading segmentation model...")
        state_dict = torch.load(config["weakseg_model_save_path"], map_location=DEVICE, weights_only=True)
        segmentation_model.load_state_dict(state_dict)
        segmentation_model.to(DEVICE)
        print("[Segmentation model loaded]")
    
    #####################################
    # 5. Evaluate and visualize results #
    #####################################
    # 5.1 Evaluate segmentation model
    metrics = evaluate_model(model=segmentation_model, test_loader=test_loader, threshold=0.5, device=DEVICE)
    clear_cuda_cache()
    save_metrics_to_csv(metrics, save_path=config["segmentation_metrics_save_path"])

if __name__ == "__main__":
    main()
