import torch

from src.metrics import evaluate_model
from src.dataset import load_data_wrapper
from src.fit import train_segmentation_model
from src.models import select_segmentation_model
from src.visualization import visualize_predictions
from src.utils import load_device, clear_cuda_cache, parse_args


def main():
    # Load configuration from .json config file (Baseline experiment)
    config = parse_args(expriment_name="Baseline")
    
    # Set device and clear CUDA cache
    DEVICE = load_device()
    clear_cuda_cache()
    
    # 1. Load dataset (train, val, and test loaders)
    train_loader, val_loader, test_loader = load_data_wrapper(config=config)
    
    # 2. Train baseline segmentation model (fully supervised)
    # 2.1. Select and init baseline segmentation model
    baseline_segmentation_model = select_segmentation_model(model_name=config["baseline_seg_model_name"])
    if baseline_segmentation_model is None:
        return
    baseline_segmentation_model.to(DEVICE)
    
    # 2.2. Train or load baseline segmentation model
    if config["train_segmentation"]:
        print("\n----Training baseline segmentation model...")
        baseline_segmentation_model = train_segmentation_model(model=baseline_segmentation_model,
                                                               train_loader=train_loader,
                                                               val_loader=val_loader,
                                                               config=config,
                                                               device=DEVICE
                                                               )
        torch.save(baseline_segmentation_model.state_dict(), config["baseline_seg_model_save_path"])
        clear_cuda_cache()
    else:
        print("\n----Loading baseline segmentation model...")
        state_dict = torch.load(config["baseline_seg_model_save_path"], map_location=DEVICE, weights_only=True)
        baseline_segmentation_model.load_state_dict(state_dict)
        baseline_segmentation_model.to(DEVICE)
        print("[Baseline segmentation model loaded]")
    
    # 3. Evaluate and visualize results
    evaluate_model(model=baseline_segmentation_model, test_loader=test_loader, device=DEVICE)
    visualize_predictions(model=baseline_segmentation_model, test_loader=test_loader, device=DEVICE) # TODO (Paul): add saving path in cfg
    clear_cuda_cache()


if __name__ == "__main__":
    main()