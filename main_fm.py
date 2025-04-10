# TODO (Carl): implement main weakseg with foundation models

# To run this we need to install the library for the segment anything model with:
# pip install git+https://github.com/facebookresearch/sam2.git

import torch
import os
import requests
import numpy as np
from src.utils import load_device, clear_cuda_cache
from src.fm_utils import parse_args, evaluate_and_visualise_sam
from src.dataset import load_data_wrapper

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
    # TO DO: The results are not really good right now, we should try to load the data without any applied transformation.
    # So, modify the load_data_wrapper
    train_loader, val_loader, test_loader = load_data_wrapper(config=config)
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
    
    class SAMWrapper:
        def __init__(self, predictor, device):
            self.predictor = predictor
            self.device = device

        def __call__(self, images):
            """
            Processes a batch of images using the SAM2 predictor.
            Expects images as a batch of tensors (C, H, W). Internally loops per image.
            """
            outputs = []
            for image in images:
                # Convert tensor (C,H,W) to a NumPy image (H,W,C) and scale to 0-255.
                image_np = image.permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                self.predictor.set_image(image_np)
                
                # Define a weak prompt: use the image center dynamically.
                H, W, _ = image_np.shape
                input_point = np.array([[W // 2, H // 2]])  # Note the order: (x, y)
                input_label = np.array([1])
                
                masks, _, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )
                
                outputs.append(torch.from_numpy(masks[0]).float().to(self.device))
            return torch.stack(outputs, dim=0)

        def eval(self):
            return self
    
    segmentation_model = SAMWrapper(predictor, DEVICE)
    
    #####################################
    # 3. Evaluate and visualize results #
    #####################################
    # Instead of evaluating metrics and visualizing in two separate stages,
    # process each image as it is predicted.
    evaluate_and_visualise_sam(model=segmentation_model, 
                               dataloader=test_loader, 
                               device=DEVICE,
                               threshold=0.8,
                               n_samples=1)
    
    clear_cuda_cache()

if __name__ == "__main__":
    main()
