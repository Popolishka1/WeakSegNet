# TODO (Carl): implement main weakseg with foundation models

# To run this we need to install the library for the segment anything model with:
# pip install -q segment-anything-py

import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_device, clear_cuda_cache
from src.fm_utils import parse_args
from src.dataset import load_data_wrapper
from src.metrics import evaluate_model, save_metrics_to_csv
from src.fm_utils import visualise_predictions_sam

from segment_anything import SamPredictor, sam_model_registry



def main():
    # Load configuration from .json config file
    config = parse_args(experiment_name="SAM")
    print("Loaded the configs.\n")
    
    # Set device and clear CUDA cache
    DEVICE = load_device()
    clear_cuda_cache()
    
    ##################################################
    # 1. Load dataset (train, val, and test loaders) #
    ##################################################
    train_loader, val_loader, test_loader = load_data_wrapper(config=config)
    print("Created data loaders.\n")
    
    ##################################################
    # 2. Download and initialise SAM                #
    ##################################################
    
    # Download SAM ViT-B checkpoint if needed
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    response = requests.get(url)
    with open("sam_vit_b_01ec64.pth", "wb") as f:
        f.write(response.content)
    print("Checkpoint downloaded successfully.")
    
    # Initialise SAM and create the SAMWrapper
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    
    class SAMWrapper:
        def __init__(self, predictor):
            self.predictor = predictor

        def __call__(self, images):
            """
            Processes a batch of images using the SAM predictor.
            Expects images as a batch of tensors (C, H, W). Internally loops per image.
            """
            outputs = []
            for image in images:
                # Convert tensor (C,H,W) to a NumPy image (H,W,C) and scale 0-255.
                image_np = image.permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                self.predictor.set_image(image_np)
                
                # Define a default prompt (for instance, the image center)
                input_point = np.array([[112, 112]])
                input_label = np.array([1])
                
                masks, _, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=None,
                    multimask_output=False
                )
                outputs.append(torch.from_numpy(masks[0]).float())
            return torch.stack(outputs, dim=0)

        def eval(self):
            return self
    
    segmentation_model = SAMWrapper(predictor)
    
    #####################################
    # 3. Evaluate and visualize results #
    #####################################
    
    # Evaluate segmentation model (this calls your existing evaluate_model,
    # which expects a model that can process a batch of images)
    metrics = evaluate_model(model=segmentation_model, test_loader=test_loader, threshold=0.5, device=DEVICE)
    clear_cuda_cache()
    save_metrics_to_csv(metrics, save_path=config["segmentation_metrics_save_path"])
    
    # Visualise predictions (using our SAM-specific visualisation function)
    visualise_predictions_sam(config=config,
                              dataloader=test_loader,
                              model=segmentation_model,
                              n_samples=5,
                              threshold=0.5,
                              device=DEVICE)
    clear_cuda_cache()
    
if __name__ == "__main__":
    main()
