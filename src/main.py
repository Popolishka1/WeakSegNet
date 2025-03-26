import os
import torch
import torch.nn as nn
from fit import fit_adam
from baseline import PetUNet
from dataset import data_loading
from metrics import evaluate_model
from visualization import visualize_predictions
from utilities import load_device, clear_cuda_cache

# TODO: create some different wrapup functions for: 1. baseline computation 2. weakseg training
 
if __name__ == "__main__":

    TRAIN = True # Set to True if you wish to train

    DEVICE = load_device()
    clear_cuda_cache()

    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..")) # This is the repo directory: WeakSegNet/
    FILE_PATH = os.path.join(BASE_DIR, "data") # This the directory of the data folder (with the annotations and images subfolders)
    
    #################
    # Load the data #
    #################
    image_size = 256
    batch_size_train, batch_size_val, batch_size_test = 1, 1, 1
    val_split = 0.2
    FULLY_SUPERVISED = True
    WEAKLY_SUPERVISED = False

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = data_loading(
                                                                        path=FILE_PATH,
                                                                        fully_supervised=FULLY_SUPERVISED,
                                                                        weakly_supervised=WEAKLY_SUPERVISED,
                                                                        image_size=image_size,
                                                                        batch_size_train=batch_size_train,
                                                                        batch_size_val=batch_size_val,
                                                                        batch_size_test=batch_size_test,
                                                                        val_split = val_split
                                                                        )

    if TRAIN:
        #################
        # Fit the model # 
        #################
        baseline_model = PetUNet(in_channels=3, out_channels=1).to(DEVICE)

        loss_fn = nn.BCELoss()
        n_epochs = 25
        lr = 0.0001

        fit_adam(model=baseline_model,
                loss_fn=loss_fn,
                n_epochs=n_epochs,
                lr=lr,
                train_loader=train_loader,
                val_loader=val_loader,
                device=DEVICE
        )
        baseline_model.to(torch.device("cpu"))
        torch.save(baseline_model.state_dict(), "baseline_pet_unet.pth")
        clear_cuda_cache()

        ###########################
        # Evaluation of the model #
        ###########################
        baseline_model.to(DEVICE)
        dice_score, avg_accuracy = evaluate_model(model=baseline_model, test_loader=test_loader, device=DEVICE)
        clear_cuda_cache()

        ################################
        # Visualization of the results #
        ################################
        visualize_predictions(model=baseline_model, test_loader=test_loader, device=DEVICE)
        clear_cuda_cache()

    else:
        ###########################
        # Evaluation of the model #
        ###########################
        baseline_model = PetUNet()
        state_dict = torch.load("baseline_pet_unet.pth", map_location=DEVICE, weights_only=True)
        baseline_model.load_state_dict(state_dict)
        baseline_model.to(DEVICE)

        avg_dice, avg_accuracy = evaluate_model(model=baseline_model, test_loader=test_loader, device=DEVICE)
        clear_cuda_cache()

        ################################
        # Visualization of the results #
        ################################
        visualize_predictions(model=baseline_model, test_loader=test_loader, device=DEVICE)
        clear_cuda_cache()