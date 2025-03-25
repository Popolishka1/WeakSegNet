import os
import torch
import torch.nn as nn
from fit import fit_adam
from baseline import PetUNet
from metrics import evaluate_model
from data_loading import data_loading


if __name__ == "__main__":

    TRAIN = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Using device: {DEVICE}]")

    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..")) # This is the repo directory: WeakSegNet/
    FILE_PATH = os.path.join(BASE_DIR, "data", )

    # Load the data 
    batch_size_train, batch_size_test = 16, 16
    image_size = 256
    train_dataset, test_dataset, trainloader, testloader = data_loading(path=FILE_PATH,
                                                                        image_size=image_size,
                                                                        batch_size_train=batch_size_train,
                                                                        batch_size_test=batch_size_test
                                                                        )
    
    if TRAIN:
        # Fit the model
        model = PetUNet(in_channels=3, out_channels=1).to(DEVICE)

        loss_fn = nn.BCELoss()
        n_epochs = 10
        lr = 0.0001

        fit_adam(model=model, loss_fn=loss_fn, n_epochs=n_epochs, lr=lr, trainloader=trainloader, device=DEVICE)
        torch.save(model.state_dict(), "baseline_pet_unet.pth")

        # Evaluation of the model
        model.to(DEVICE)
        dice_score = evaluate_model(model=model, testloader=testloader, device=DEVICE) 

    else:
        model = PetUNet().to(DEVICE)
        model.load_state_dict(torch.load("baseline_pet_unet.pth"))
