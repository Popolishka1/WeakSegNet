# TODO (Carl): implement main weakseg with foundation models

# To run this we need to install the library for the segment anything model with:
# pip install -q segment-anything-py

import torch
import torchvision
from src.fm_utils import load_device, clear_cuda_cache, parse_args
from src.dataset import load_data_wrapper
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import requests
import numpy as np


def main():
    # Load configuration from .json config file (CAM experiment)
    config = parse_args(experiment_name="SAM")
    print("Loaded the configs. \n")
    
    # Set device and clear CUDA cache
    DEVICE = load_device()
    clear_cuda_cache()
    
    ##################################################
    # 1. Load dataset (train, val, and test loaders) #
    ##################################################
    train_loader, val_loader, test_loader = load_data_wrapper(config=config)  # If doesn't work check that the transformation in there is consistent with what SAM expects
    print("Created data loaders. \n")
    
    ##################################################
    # 2. Use SAM to perform segmentation             #
    ##################################################
    
    # download SAM ViT-B checkpoint
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    response = requests.get(url)
    with open("sam_vit_b_01ec64.pth", "wb") as f:
        f.write(response.content)

    print("Checkpoint downloaded successfully.")

    # initialise the model + predictor
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(DEVICE)
    predictor = SamPredictor(sam)


    def sam_masker(image):
        """
        SAM only segmentation
        """
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # set image in SAM predictor
        predictor.set_image(image_np)

        # region of interest (default centre)
        input_point = np.array([[112, 112]])
        input_label = np.array([1])

        # SAM to predict mask
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,                           # No bounding box (TODO)
            multimask_output=False
        )

        return torch.from_numpy(masks[0]).float()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            SAM_masks = torch.stack([sam_masker(img) for img in images]).to(DEVICE)

            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            img = images[0].cpu().permute(1, 2, 0).numpy()
            sams = SAM_masks[0].cpu().numpy()

            ax[0].imshow(img)
            ax[0].set_title("Input Image")
            ax[0].axis("off")

            ax[1].imshow(img)
            ax[1].imshow(sams, alpha=0.5)
            ax[1].set_title("SAM Mask")
            ax[1].axis("off")

            plt.show()

if __name__ == "__main__":
    main()
