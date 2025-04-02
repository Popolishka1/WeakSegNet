import os
import json
import torch
import argparse
import warnings
from torch.utils.data import DataLoader

from src.models import UNet, DeepLabV3, FCN
from src.cam_utils import generate_pseudo_masks
from src.utils import load_device, clear_cuda_cache
from src.dataset import data_loading, PseudoMaskDataset
from src.fit import fit_segmentation, fit_classification
from src.metrics import evaluate_model, evaluate_classifier
from src.classification import ResNet50, DenseNet121, ResNet101
from src.visualization import visualize_predictions, visualize_cams


def parse_args():
    parser = argparse.ArgumentParser(description="Weak supervision segmentation training and evaluation (CAM use case)")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to .json config file containing all experiment settings"
                        )
    args = parser.parse_args()
    # Load the configuration from the .json file
    with open(args.config, "r") as f:
        config = json.load(f)
    return config


def load_data_wrapper(config):
    BASE_DIR = os.getcwd()
    FILE_PATH = os.path.join(BASE_DIR, config["data_folder"])
    
    # Data split and batch sizes from the config
    batch_size_train = config["train_batch_size"]
    batch_size_val = config["val_batch_size"]
    batch_size_test = config["test_batch_size"]
    val_split = config["val_split"]
    data_split_size = (batch_size_train, batch_size_val, batch_size_test, val_split)
    
    # Image size for resize  
    image_size = config["image_size"]
    
    return data_loading(path=FILE_PATH, data_split_size=data_split_size, image_size=image_size)


def select_classifier(model_name, n_classes):
    if model_name == "ResNet50":
        return ResNet50(n_classes=n_classes)
    if model_name == "ResNet101":
        return ResNet101(n_classes=n_classes)
    if model_name == "DenseNet12":
        return DenseNet121(n_classes=n_classes)
    else:
        warnings.warn("Incorrect classifier name or model not implemented.")
        return None


def train_classifier(classifier, train_loader, val_loader, config, device):
    n_epochs = config["n_epochs_cls"]
    lr = config["learning_rate_cls"]
    fit_classification(classifier=classifier,
             n_epochs=n_epochs,
             lr=lr,
             train_loader=train_loader,
             val_loader=val_loader,
             device=device)
    return classifier


def select_segmentation_model(model_name):
    if model_name == "UNet":
        return UNet()
    elif model_name == "DeepLabV3":
        return DeepLabV3()
    elif model_name == "FCN":
        return FCN()
    else:
        warnings.warn("Incorrect model name or model not implemented.")
        return None


def train_segmentation_model(model, train_loader, val_loader, config, device):
    n_epochs = config["n_epochs_seg"]
    lr = config["learning_rate_seg"]
    fit_segmentation(model=model,
             n_epochs=n_epochs,
             lr=lr,
             train_loader=train_loader,
             val_loader=val_loader,
             device=device)
    return model
 

def main():
    # Load configuration from .json config file
    config = parse_args()
    
    # Set device and clear CUDA cache
    device = load_device()
    clear_cuda_cache()
    
    # 1. Load dataset (train, val, and test loaders)
    train_loader, val_loader, test_loader = load_data_wrapper(config)

    # 2. Train classifier
    # 2.1. Select and init classifier
    classifier = select_classifier(config["classifier"], config["n_classes"])
    if classifier is None:
        return
    classifier.to(device)

    # 2.2. Train or load classifier
    if config["train_classifier"]:
        print("\n----Training classifier...")
        classifier = train_classifier(classifier,
                                      train_loader,
                                      val_loader,
                                      config,
                                      device
                                      )
        torch.save(classifier.state_dict(), config["classifier_save_path"])
        clear_cuda_cache()

    else:
        print("\n----Loading classifier...")
        state_dict = torch.load(config["classifier_save_path"], map_location=device, weights_only=True)
        classifier.load_state_dict(state_dict)
        classifier.to(device)
        print("[Classifier loaded]")

    # 2.3. Evaluate classifier
    evaluate_classifier(classifier=classifier, test_loader=test_loader, device=device)

    # 3. Generate pseudo masks
    cam_threshold = config["cam_threshold"]
    print(f"\n----Generating pseudo masks from CAM with cam_threshold={cam_threshold}...")
    pseudo_masks = generate_pseudo_masks(classifier=classifier,
                                         dataloader=train_loader,
                                         cam_threshold=cam_threshold,
                                         device=device
                                        )
    # TODO: refine CAM with dense CRF for instance to get better pseudo masks
    train_dataset_pseudo = PseudoMaskDataset(train_loader.dataset, pseudo_masks)
    train_loader_pseudo = DataLoader(train_dataset_pseudo,  batch_size=config["train_batch_size"], shuffle=True, num_workers=4)
    assert len(pseudo_masks) == len(train_loader.dataset), "Mismatched pseudo masks!"
    print(f"[Generated {len(pseudo_masks)} pseudo masks]")

    visualize_cams(classifier=classifier, test_loader=test_loader, device=device, cam_threshold=cam_threshold, save_path="cam_examples.png")

    # 4. Train segmentation model (weakly supervised)
    # 4.1 Select and init baseline segmentation model
    segmentation_model = select_segmentation_model(config["segmentation_model"])
    if segmentation_model is None:
        return
    segmentation_model.to(device)

    # 4.2 Train or load baseline segmentation model
    if config["train_segmentation"]:
        print("\n----Training segmentation model...")
        segmentation_model = train_segmentation_model(segmentation_model,
                                                      train_loader_pseudo,
                                                      val_loader,
                                                      config,
                                                      device)
        torch.save(segmentation_model.state_dict(), config["weakseg_model_save_path"])
        clear_cuda_cache()
    else:
        print("\n----Loading segmentation model...")
        state_dict = torch.load(config["weakseg_model_save_path"], map_location=device, weights_only=True)
        segmentation_model.load_state_dict(state_dict)
        segmentation_model.to(device)
        print("[Segmentation model loaded]")


    # 5. Evaluate and visualize results
    evaluate_model(model=segmentation_model, test_loader=test_loader, device=device)
    visualize_predictions(model=segmentation_model, test_loader=test_loader, device=device) # TODO: add saving path in cfg
    clear_cuda_cache()


if __name__ == "__main__":
    main()