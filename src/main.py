import os
import torch
import torch.nn as nn
from fit import fit_adam
from baseline import PetUNet
from dataset import data_loading
from metrics import evaluate_model
from utilities import load_device, clear_cuda_cache

if __name__ == "__main__":

    TRAIN = True # Set to True if you wish to train

    DEVICE = load_device()
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..")) # This is the repo directory: WeakSegNet/
    FILE_PATH = os.path.join(BASE_DIR, "data", ) # This the directory of the data folder (with the annotations and images subfolders)

    #################
    # Load the data #
    #################
    image_size = 256
    batch_size_train, batch_size_val, batch_size_test = 16, 16, 16
    val_split = 0.2
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = data_loading(
                                                                        path=FILE_PATH,
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
        clear_cuda_cache()
        baseline_model = PetUNet(in_channels=3, out_channels=1).to(DEVICE)

        loss_fn = nn.BCELoss()
        n_epochs = 10
        lr = 0.0001

        fit_adam(model=baseline_model,
                loss_fn=loss_fn,
                n_epochs=n_epochs,
                lr=lr,
                trainloader=train_loader,
                valloader=val_loader,
                device=DEVICE
            )
        torch.save(baseline_model.state_dict(), "baseline_pet_unet.pth")

        ###########################
        # Evaluation of the model #
        ###########################
        clear_cuda_cache()
        baseline_model.to(DEVICE)
        dice_score = evaluate_model(model=baseline_model, testloader=test_loader, device=DEVICE) 

    else:
        clear_cuda_cache()
        baseline_model = PetUNet().to(DEVICE)
        baseline_model.load_state_dict(torch.load("baseline_pet_unet.pth"))
