# TODO (Carl): generate weak mask from foundations models
import json
import torch
import argparse
import matplotlib.pyplot as plt


def parse_args(experiment_name: str):
    if experiment_name == "SAM":
        parser = argparse.ArgumentParser(description="SAM segmentation with foundation models (weakly supervised use case)")
    else:
        raise ValueError("Experiment name not recognized. Consider switching to the non foundation models utils.")
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