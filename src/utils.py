import json
import torch
import argparse
import warnings


def parse_args(expriment_name: str):
    if expriment_name == "Baseline":
        parser = argparse.ArgumentParser(description="Baseline training and evaluation (fully supervised use case)")
    elif expriment_name == "CAM":
        parser = argparse.ArgumentParser(description="CAM training and evaluation (weakly supervised use case)")
    elif expriment_name == "SAM":
        parser = argparse.ArgumentParser(description="SAM segmentation with foundation models (weakly supervised use case)")
    else:
        warnings.warn("Experiment name not recognized. Defaulting to Baseline.")
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


def load_device():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Using device: {DEVICE}]")
    return DEVICE


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()