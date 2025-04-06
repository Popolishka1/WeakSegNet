import torch
from torch.utils.data import DataLoader

from src.cam_utils import generate_pseudo_masks
from src.classification import select_classifier
from src.models import select_segmentation_model
from src.metrics import evaluate_model, evaluate_classifier
from src.dataset import load_data_wrapper, PseudoMaskDataset
from src.fit import train_classifier, train_segmentation_model
from src.utils import load_device, clear_cuda_cache, parse_args
from src.visualisation import visualise_predictions, visualise_cams


def main():
    # Load configuration from .json config file (CAM experiment)
    config = parse_args(expriment_name="CAM")
    
    # Set device and clear CUDA cache
    DEVICE = load_device()
    clear_cuda_cache()
    
    ##################################################
    # 1. Load dataset (train, val, and test loaders) #
    ##################################################
    train_loader, val_loader, test_loader = load_data_wrapper(config=config)

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

    ############################
    # 3. Generate pseudo masks #
    ############################
    # 3.1 Generate pseudo masks using CAM
    cam_threshold = config["cam_threshold"]
    cam_type = config["cam_model"]
    print(f"\n----Generating pseudo masks from {cam_type} with cam_threshold={cam_threshold}...")

    pseudo_masks = generate_pseudo_masks(dataloader=train_loader,
                                         classifier=classifier,
                                         cam_type=cam_type,
                                         cam_threshold=cam_threshold,
                                         device=DEVICE
                                        )
    clear_cuda_cache()
    
    # Create a new train dataset with pseudo masks
    train_dataset_pseudo = PseudoMaskDataset(original_dataset=train_loader.dataset,
                                             pseudo_masks=pseudo_masks
                                             )
    train_loader_pseudo = DataLoader(train_dataset_pseudo, batch_size=config["train_batch_size"], shuffle=True, num_workers=4)
    assert len(pseudo_masks) == len(train_loader.dataset), "Mismatched pseudo masks!"
    print(f"[Generated {len(pseudo_masks)} pseudo masks]")

    # 3.2 Visualise CAMs
    visualise_cams(config=config,
                   dataloader=train_loader_pseudo,
                   classifier=classifier,
                   cam_type=cam_type,
                   cam_threshold=cam_threshold,
                   n_samples=5,
                   device=DEVICE
                   )
    clear_cuda_cache()

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
    evaluate_model(model=segmentation_model, test_loader=test_loader, threshold=0.5, device=DEVICE)
    clear_cuda_cache()

    # 5.2 Visualise predictions
    visualise_predictions(config=config,
                          dataloader=test_loader,
                          model=segmentation_model,
                          n_samples=5,
                          threshold=0.5,
                          device=DEVICE
                          )
    clear_cuda_cache()


if __name__ == "__main__":
    main()