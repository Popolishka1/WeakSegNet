import torch
from torch.utils.data import DataLoader

from src.cam_utils import generate_pseudo_masks
from src.classification import select_classifier
from src.models import select_segmentation_model
from src.metrics import evaluate_model, evaluate_classifier
from src.dataset import load_data_wrapper, PseudoMaskDataset
from src.fit import train_classifier, train_segmentation_model
from src.utils import load_device, clear_cuda_cache, parse_args
from src.visualization import visualize_predictions, visualize_cams


def main():
    # Load configuration from .json config file (CAM experiment)
    config = parse_args(expriment_name="CAM")
    
    # Set device and clear CUDA cache
    DEVICE = load_device()
    clear_cuda_cache()
    
    # 1. Load dataset (train, val, and test loaders)
    train_loader, val_loader, test_loader = load_data_wrapper(config=config)

    # 2. Train classifier
    # 2.1. Select and init classifier
    # TODO (Paul): wrap up
    classifier = select_classifier(model_name=config["classifier"], n_classes=config["n_classes"])
    if classifier is None:
        return
    classifier.to(DEVICE)

    # 2.2. Train or load classifier
    # TODO (Paul): wrap up
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

    # 3. Generate pseudo masks
    # TODO (Paul): wrap up the mask gen + viz
    cam_threshold = config["cam_threshold"]
    cam_model = config["cam_model"]
    print(f"\n----Generating pseudo masks from {cam_model} with cam_threshold={cam_threshold}...")
    pseudo_masks = generate_pseudo_masks(classifier=classifier,
                                         dataloader=train_loader,
                                         cam_threshold=cam_threshold,
                                         device=DEVICE, 
                                         model=cam_model
                                        )

    train_dataset_pseudo = PseudoMaskDataset(original_dataset=train_loader.dataset,
                                             pseudo_masks=pseudo_masks
                                             )
    train_loader_pseudo = DataLoader(train_dataset_pseudo, batch_size=config["train_batch_size"], shuffle=True, num_workers=4)
    assert len(pseudo_masks) == len(train_loader.dataset), "Mismatched pseudo masks!"
    print(f"[Generated {len(pseudo_masks)} pseudo masks]")

    visualize_cams(classifier=classifier,
                   test_loader=test_loader,
                   device=DEVICE,
                   cam_threshold=cam_threshold,
                   save_path="cam_examples.png"
                   ) # TODO (Paul): add saving path in cfg

    # 4. Train segmentation model (weakly supervised)
    # TODO (Paul): wrap up 
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


    # 5. Evaluate and visualize results
    evaluate_model(model=segmentation_model, test_loader=test_loader, device=DEVICE)
    visualize_predictions(model=segmentation_model, test_loader=test_loader, device=DEVICE) # TODO (Paul): add saving path in cfg
    clear_cuda_cache()


if __name__ == "__main__":
    main()