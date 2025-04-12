# TODO (Carl): implement main weakseg with foundation models

# To run this we need to install the library for the segment anything model with:
# pip install -q segment-anything-py

import torch
import torchvision
from src.fm_utils import load_device, clear_cuda_cache, parse_args
from src.dataset import load_data_wrapper
from PIL import Image, ImageDraw
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

        img_tensor = images[0].cpu()
        sam_mask = SAM_masks[0].cpu().numpy()

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)  
        img_pil = Image.fromarray(img_np)

        sam_mask = sam_mask.squeeze().astype(np.uint8) 

        overlay = Image.new('RGBA', img_pil.size, (255, 0, 0, 0))
        alpha = Image.fromarray((sam_mask * 128).astype(np.uint8))
        overlay.putalpha(alpha)

        combined = Image.alpha_composite(img_pil.convert('RGBA'), overlay)

        total_width = img_pil.width * 2
        combined_image = Image.new('RGB', (total_width, img_pil.height))
        combined_image.paste(img_pil, (0, 0))
        combined_image.paste(combined.convert('RGB'), (img_pil.width, 0))

        combined_image.show()


if __name__ == "__main__":
    main()
