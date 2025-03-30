import os
import torch
import warnings
import torch.nn as nn
from fit import fit_adam
from metrics import evaluate_model
from models import UNet, DeepLabV3, FCN
from visualization import visualize_predictions
from dataset import data_loading, data_transform
from utilities import load_device, clear_cuda_cache


def model(model_name: str):
    if model_name == "UNet":
        return UNet()
    if model_name == "DeepLabV3":
        return  DeepLabV3()
    if model_name == "FCN":
        return FCN()
    else:
        warnings.warn("ncorrect baseline name or model not implemented.") 


if __name__ == "__main__":

    TRAIN = True # Set to True if you wish to train

    DEVICE = load_device()
    clear_cuda_cache()

    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..")) # This is the repo directory: WeakSegNet/
    FILE_PATH = os.path.join(BASE_DIR, "data") # This the directory of the data folder (with the annotations and images subfolders)
    
    #################
    # Load the data #
    #################

    # Data splits: configure the size of the splits
    batch_size_train, batch_size_val, batch_size_test = 64, 32, 64
    val_split = 0.2 # TODO: consider increasing
    size = (batch_size_train, batch_size_val, batch_size_test, val_split)

    # Configure which data you wish to load
    FULLY_SUPERVISED = True
    WEAKLY_SUPERVISED = False
    WEAKLY_SUPERVISED_BBOX_TO_MASK_DUMMY = False
    experiment = (FULLY_SUPERVISED, WEAKLY_SUPERVISED, WEAKLY_SUPERVISED_BBOX_TO_MASK_DUMMY)

    # Configure which data transformation you wish to load
    image_size = 256
    image_transform, mask_transform = data_transform(image_size=image_size)

    _, _, _, train_loader, val_loader, test_loader = data_loading(path=FILE_PATH,
                                                                  experiment=experiment,
                                                                  image_transform=image_transform,
                                                                  mask_transform=mask_transform,
                                                                  size=size
                                                                )

    MODEL = "UNet"

    if TRAIN:
        #################
        # Fit the model # 
        #################

        model_weakseg = model(model=MODEL, device=DEVICE)
        model_weakseg.to(DEVICE)

        loss_fn = nn.BCELoss()
        n_epochs = 1
        lr = 0.0001

        fit_adam(model=model_weakseg,
                loss_fn=loss_fn,
                n_epochs=n_epochs,
                lr=lr,
                train_loader=train_loader,
                val_loader=val_loader,
                device=DEVICE
        )
        model_weakseg.to(torch.device("cpu"))
        torch.save(model_weakseg.state_dict(), "test.pth")
        clear_cuda_cache()

        ###########################
        # Evaluation of the model #
        ###########################
        model_weakseg.to(DEVICE)
        dice_score, avg_accuracy = evaluate_model(model=model_weakseg, test_loader=test_loader, device=DEVICE)
        clear_cuda_cache()

        ################################
        # Visualization of the results #
        ################################
        visualize_predictions(model=model_weakseg, test_loader=test_loader, device=DEVICE)
        clear_cuda_cache()

    else:
        ###########################
        # Evaluation of the model #
        ###########################
        model_weakseg = model(baseline_name=MODEL, device=DEVICE)

        state_dict = torch.load("test.pth", map_location=DEVICE, weights_only=True)
        model_weakseg.load_state_dict(state_dict)
        model_weakseg.to(DEVICE)

        avg_dice, avg_accuracy = evaluate_model(model=model_weakseg, test_loader=test_loader, device=DEVICE)
        clear_cuda_cache()

        ################################
        # Visualization of the results #
        ################################
        visualize_predictions(model=model_weakseg, test_loader=test_loader, device=DEVICE)
        clear_cuda_cache()