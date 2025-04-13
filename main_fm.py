# To run this we need to install the library for the segment anything model with:
# pip install git+https://github.com/facebookresearch/sam2.git

import torch
import os
import requests
import numpy as np
from torch.utils.data import DataLoader

from src.cam_utils import generate_pseudo_masks, get_cam_generator
from src.classification import select_classifier
from src.models import select_segmentation_model
from src.dataset import load_data_wrapper, PseudoMaskDataset
from src.fit import train_classifier, train_segmentation_model
from src.utils import load_device, clear_cuda_cache, parse_args
from src.visualisation import visualise_predictions, visualise_cams
from src.metrics import evaluate_model, evaluate_classifier, save_metrics_to_csv
from src.fm_utils import SAMWrapper, SAMWrapperWithCAM

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def zero_shot_SAM():
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
    train_loader, val_loader, test_loader = load_data_wrapper(config=config, sam=SAM_FLAG)
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
    metrics = evaluate_model(segmentation_model, test_loader, threshold=0.5, device=DEVICE)
    print("Final test set evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    save_metrics_to_csv(metrics, save_path=config["segmentation_metrics_save_path"])
    
    clear_cuda_cache()
    
    # 5. Visualize predictions.
    visualise_predictions(config, test_loader, segmentation_model,
                          n_samples=5, threshold=0.5, device=DEVICE)
    
    clear_cuda_cache()



def cam_pseudo_mask_training():
    """
    Function that implements training a segmentation model using CAM-generated pseudo masks.
    This integrates classifier training, CAM-based pseudo mask generation, and segmentation model training.
    """
    # Load configuration from .json config file
    config = parse_args(expriment_name="CAM_PseudoMask")
    print("Loaded the configs.\n")
    
    # Set device and clear CUDA cache
    DEVICE = load_device()
    clear_cuda_cache()
    
    # Set sam flag to True for minimal transformations
    SAM_FLAG = True
    
    ##################################################
    # 1. Load dataset (train, val, and test loaders) #
    ##################################################
    train_loader, val_loader, test_loader = load_data_wrapper(config=config, sam=SAM_FLAG)
    print("Created data loaders.\n")

    #######################
    # 2. Train classifier #
    #######################
    # 2.1. Select and init classifier
    classifier = select_classifier(model_name=config["classifier"], n_classes=config["n_classes"])
    if classifier is None:
        return
    classifier.to(DEVICE)

    # 2.2. Train or load classifier
    if config["train_classifier"]:
        print("\n----Training classifier...")
        classifier = train_classifier(classifier=classifier,
                                     train_loader=train_loader,
                                     val_loader=val_loader,
                                     config=config,
                                     device=DEVICE
                                     )
        torch.save(classifier.state_dict(), config["classifier_save_path"])
        clear_cuda_cache()
    else:
        print("\n----Loading classifier...")
        state_dict = torch.load(config["classifier_save_path"], map_location=DEVICE, weights_only=True)
        classifier.load_state_dict(state_dict)
        classifier.to(DEVICE)
        print("[Classifier loaded]")

    # 2.3. Evaluate classifier
    evaluate_classifier(classifier=classifier, test_loader=test_loader, config=config, device=DEVICE)
    
    ###############################
    # 3. Generate CAM pseudo masks #
    ###############################
    cam_threshold = config["cam_threshold"]
    cam_type = config["cam_model"]

    print(f"\n----Generating pseudo masks from {cam_type} with cam_threshold={cam_threshold}...")
    pseudo_masks = generate_pseudo_masks(
        dataloader=train_loader,
        classifier=classifier,
        cam_type=cam_type,
        cam_threshold=cam_threshold,
        device=DEVICE
    )

    ###################################
    # 4. Generate CAMSAM pseudo masks #
    ###################################
    sam2_checkpoint = "sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    if not os.path.exists(sam2_checkpoint):
        print(f"Checkpoint file not found at {sam2_checkpoint}. Downloading...")
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        response = requests.get(url, stream=True)
        with open(sam2_checkpoint, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed!")

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    
    camsam = SAMWrapperWithCAM(
        predictor=predictor,
        pseudo_masks=pseudo_masks,
        num_points=config["num_points"],
        device=DEVICE
    )
    
    pseudo_masks = []
    original_dataset = train_loader.dataset
    print("Generating pseudo masks using SAM:")
    for idx in range(len(original_dataset)):
        image, _, info = original_dataset[idx]
        image = image.to(DEVICE).unsqueeze(0)
    
        with torch.no_grad():
            mask = camsam(image)
        pseudo_mask = (mask > 0.5).float().squeeze(0)
        pseudo_masks.append(pseudo_mask)
    
        if idx % 10 == 0:
            print(f"Processed {idx + 1}/{len(original_dataset)} images.")
    print("Pseudo mask generation complete.\n")
    
    
    ###################################################
    # 4. Create pseudo mask dataset and dataloader    #
    ###################################################
    original_dataset = train_loader.dataset
    train_dataset_pseudo = PseudoMaskDataset(original_dataset, pseudo_masks)
    train_loader_pseudo = DataLoader(
        train_dataset_pseudo, 
        batch_size=config["train_batch_size"], 
        shuffle=True, 
        num_workers=0
    )
    print("Created pseudo mask dataset and dataloader.\n")
    
    ###################################################
    # 5. Train segmentation model (weakly supervised) #
    ###################################################
    # 5.1 Select and init segmentation model
    segmentation_model = select_segmentation_model(config["segmentation_model"])
    if segmentation_model is None:
        return
    segmentation_model.to(DEVICE)
    
    # 5.2 Train or load segmentation model
    if config["train_segmentation"]:
        print("\n----Training segmentation model with CAM pseudo masks...")
        segmentation_model = train_segmentation_model(
            model=segmentation_model,
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
    # 6. Evaluate and visualize results #
    #####################################
    # 6.1 Evaluate segmentation model
    metrics = evaluate_model(
        model=segmentation_model, 
        test_loader=test_loader, 
        threshold=0.5, 
        device=DEVICE
    )
    print("Final test set evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    save_metrics_to_csv(metrics, save_path=config["segmentation_metrics_save_path"])
    
    clear_cuda_cache()
    
    # 6.2 Visualize predictions
    visualise_predictions(
        config=config,
        dataloader=test_loader,
        model=segmentation_model,
        n_samples=5,
        threshold=0.5,
        device=DEVICE
    )
    
    clear_cuda_cache()
    
    return segmentation_model


def pseudo_mask_SAM():
    # Load configuration (make sure the experiment name is distinct if needed)
    config = parse_args(expriment_name="SAM_Pseudo")
    print("Loaded configuration for pseudo mask training.\n")
    
    # Set device and clear CUDA cache
    DEVICE = load_device()
    clear_cuda_cache()
    
    # Set the SAM flag so the minimal transformations (for SAM) are applied.
    SAM_FLAG = True
    
    ##################################################
    # 1. Load dataset (train, val, and test loaders) #
    ##################################################
    train_loader, val_loader, test_loader = load_data_wrapper(config=config, sam=SAM_FLAG)
    print("Data loaders created.\n")
    
    ##################################################
    # 2. Download and initialise SAM                #
    ##################################################
    sam2_checkpoint = "sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    if not os.path.exists(sam2_checkpoint):
        print(f"Checkpoint file not found at {sam2_checkpoint}. Downloading...")
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        response = requests.get(url, stream=True)
        with open(sam2_checkpoint, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed!")
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    sam_model = SAMWrapper(predictor, DEVICE)
    
    ############################
    # 3. Generate pseudo masks #
    ############################
    # Here, we assume that train_loader.dataset is a Subset of the OxfordPet dataset.
    # We generate a SAM-based pseudo mask for each sample.
    pseudo_masks = []
    original_dataset = train_loader.dataset  # This is typically a Subset (or a wrapped version) of OxfordPet.
    print("Generating pseudo masks using SAM:")
    for idx in range(len(original_dataset)):
        image, _, info = original_dataset[idx]  # we ignore the ground truth mask
        # Ensure the image is a tensor of shape (C, H, W) on DEVICE.
        image = image.to(DEVICE)
        # Add batch dimension: shape (1, C, H, W)
        image = image.unsqueeze(0)
        
        # Use SAM to predict a mask.
        with torch.no_grad():
            # SAMWrapper processes batches (and internally iterates over images)
            mask = sam_model(image)  # Expected shape: (1, H, W)
        # Convert SAM's output into a binary mask via thresholding (0.5 is common; adjust if needed)
        pseudo_mask = (mask > 0.5).float().squeeze(0)  # now shape is (H, W)
        pseudo_masks.append(pseudo_mask)
        if idx % 10 == 0:
            print(f"Processed {idx + 1}/{len(original_dataset)} images.")
    print("Pseudo mask generation complete.\n")

    print(f"\n----Saving generated SAM pseudo masks...")
    os.makedirs("results/sam_pseudo_masks", exist_ok=True)
    
    # Save as tensor files
    for idx, mask in enumerate(pseudo_masks):
        save_path = os.path.join("results/sam_pseudo_masks", f"pseudo_mask_{idx}.pt")
        torch.save(mask, save_path)
    
    train_dataset_pseudo = PseudoMaskDataset(original_dataset, pseudo_masks)
    train_loader_pseudo = DataLoader(train_dataset_pseudo, batch_size=config["train_batch_size"], shuffle=True, num_workers=0)
    
    ###################################################
    # 4. Train segmentation model (weakly supervised) #
    ###################################################
    # 4.1 Select and init baseline segmentation model
    segmentation_model = select_segmentation_model(config["segmentation_model"])
    if segmentation_model is None:
        return
    segmentation_model.to(DEVICE)
    
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

    # 5.2 Visualise predictions
    visualise_predictions(config=config,
                          dataloader=test_loader,
                          model=segmentation_model,
                          n_samples=5,
                          threshold=0.5,
                          device=DEVICE
                          )
    clear_cuda_cache()


def main(mode="zero_shot"):
    if mode == "zero_shot":
        zero_shot_SAM()
    elif mode == "pseudo_mask":
        pseudo_mask_SAM()
    elif mode == "cam_pseudo_mask":
        cam_pseudo_mask_training()

if __name__ == "__main__":
    main(mode = "cam_pseudo_mask")
