import os
import json
import torch
import argparse
import warnings

from src.fit import fit_segmentation
from src.dataset import data_loading
from src.metrics import evaluate_model
from src.models import UNet, DeepLabV3, FCN
from src.visualization import visualize_predictions
from src.utils import load_device, clear_cuda_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline segmentation training and evaluation (fully supervised use case)")
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


def select_baseline_segmentation_model(model_name):
    if model_name == "UNet":
        return UNet()
    elif model_name == "DeepLabV3":
        return DeepLabV3()
    elif model_name == "FCN":
        return FCN()
    else:
        warnings.warn("Incorrect baseline name or model not implemented.")
        return None


def train_model_segmentation(model, train_loader, val_loader, config, device):
    n_epochs = config["n_epochs"]
    lr = config["learning_rate"]
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
    _, _, _, train_loader, val_loader, test_loader = load_data_wrapper(config)
    
    # 2. Train baseline segmentation model (fully supervised)
    # 2.1. Select and init baseline segmentation model
    baseline_segmentation_model = select_baseline_segmentation_model(config["baseline_seg_model_name"])
    if baseline_segmentation_model is None:
        return
    baseline_segmentation_model.to(device)
    
    # 2.2. Train or load baseline segmentation model
    if config["train_segmentation"]:
        print("\n----Training baseline segmentation model...")
        baseline_segmentation_model = train_model_segmentation(baseline_segmentation_model, train_loader, val_loader, config, device)
        torch.save(baseline_segmentation_model.state_dict(), config["baseline_seg_model_save_path"])
        clear_cuda_cache()
    else:
        print("\n----Loading baseline segmentation model...")
        state_dict = torch.load(config["baseline_seg_model_save_path"], map_location=device, weights_only=True)
        baseline_segmentation_model.load_state_dict(state_dict)
        baseline_segmentation_model.to(device)
        print("[Baseline segmentation model loaded]")
    
    # 3. Evaluate and visualize results
    evaluate_model(model=baseline_segmentation_model, test_loader=test_loader, device=device)
    visualize_predictions(model=baseline_segmentation_model, test_loader=test_loader, device=device)
    clear_cuda_cache()


if __name__ == "__main__":
    main()