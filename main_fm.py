# TODO (Carl): implement main weakseg with foundation models

# To run this we need to install the library for the segment anything model with:
# pip install git+https://github.com/facebookresearch/sam2.git

import torch
import os
import requests
import numpy as np
from src.utils import load_device, clear_cuda_cache, parse_args
from src.metrics import evaluate_model, save_metrics_to_csv
from src.visualisation import visualise_predictions
from src.dataset import load_data_wrapper
from src.fm_utils import SAMWrapper

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def main():
    # Load configuration from .json config file
    config = parse_args(expriment_name="SAM")
    print("Loaded the configs.\n")
    
    # Set device and clear CUDA cache
    DEVICE = load_device()
    clear_cuda_cache()
    
    # Set sam flag to True
    SAM_FLAG = True
    
    ##################################################
    # 1. Load dataset (train, val, and test loaders) #
    ##################################################
    # TO DO: The results are not really good right now, we should try to load the data without any applied transformation.
    # So, modify the load_data_wrapper
    train_loader, val_loader, test_loader = load_data_wrapper(config=config, sam=SAM_FLAG, test_loader_size=500)
    print("Created data loaders.\n")
    
    ##################################################
    # 2. Download and initialise SAM                #
    ##################################################
    sam2_checkpoint = "sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    if not os.path.exists(sam2_checkpoint):
        print(f"Checkpoint file not found at {sam2_checkpoint}. Downloading...")
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        response = requests.get(url, stream=True)
        # Write the file in chunks in case the file is large.
        with open(sam2_checkpoint, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed!")

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
        
    segmentation_model = SAMWrapper(predictor, DEVICE)
    
    #####################################
    # 3. Evaluate and visualize results #
    #####################################
    # Instead of evaluating metrics and visualizing in two separate stages,
    # process each image as it is predicted.
    metrics = evaluate_model(segmentation_model, test_loader, threshold=0.5, device=DEVICE, sam=SAM_FLAG)
    print("Final test set evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    save_metrics_to_csv(metrics, save_path=config["segmentation_metrics_save_path"])
    
    clear_cuda_cache()
    
    # 5. Visualize predictions.
    visualise_predictions(config, test_loader, segmentation_model,
                          n_samples=5, threshold=0.5, device=DEVICE, sam=SAM_FLAG)
    
    clear_cuda_cache()

if __name__ == "__main__":
    main()
