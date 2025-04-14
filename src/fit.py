import torch
import torch.nn as nn


def fit_segmentation(model, n_epochs, lr, train_loader, val_loader, device="cuda"):
    print(f"--Fitting segmentation model with {n_epochs} epochs")

    criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # lr dynamic adjustment 

    for epoch in range(1, n_epochs + 1):

        # ------- TRAINING -------
        model.train()
        n_batches_train = len(train_loader)
        train_loss = 0.0

        for images, target_masks, _ in train_loader:
            images = images.to(device)
            target_masks = target_masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, target_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() 
        avg_train_loss = train_loss / n_batches_train

        # ------- VALIDATION -------
        model.eval()
        n_batches_val = len(val_loader)
        val_loss = 0.0

        with torch.no_grad():
            for images, gt_masks, _ in val_loader:
                images = images.to(device)
                gt_masks = gt_masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, gt_masks)

                val_loss += loss.item()
        avg_val_loss = val_loss / n_batches_val

        print(f"[Segmentation] Epoch {epoch}/{n_epochs} -> Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss) # adjust the lr
    print("[Segmentation training done]")


def train_segmentation_model(model, train_loader, val_loader, config, device):
    """Wrapper function to train the segmentation model with the given configuration of the config file."""
    n_epochs = config["n_epochs_seg"]
    lr = config["learning_rate_seg"]
    fit_segmentation(model=model,
                     n_epochs=n_epochs,
                     lr=lr,
                     train_loader=train_loader,
                     val_loader=val_loader,
                     device=device
                     )
    return model


def fit_classification(classifier, target_id, n_epochs, lr, train_loader, val_loader, device="cuda"):
    print(f"--Fitting classifier with {n_epochs} epochs")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # lr dynamic adjustment 

    for epoch in range(1, n_epochs + 1):

        # ------- TRAINING -------
        classifier.train()
        n_batches_train = len(train_loader)
        train_loss = 0.0

        for images, _, info in train_loader:
            images = images.to(device)
            labels = info[target_id].to(device).long()

            outputs = classifier(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() 
        avg_train_loss = train_loss / n_batches_train

        # ------- VALIDATION -------
        classifier.eval()
        n_batches_val = len(val_loader)
        val_loss = 0.0

        with torch.no_grad():
            for images, _, info in val_loader:
                images = images.to(device)
                labels = info[target_id].to(device).long()

                outputs = classifier(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
        avg_val_loss = val_loss / n_batches_val

        print(f"[Classifier] Epoch {epoch}/{n_epochs} -> Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss) # adjust the lr
    print("[Classifier training done]")


def train_classifier(classifier, train_loader, val_loader, config, device):
    """Wrapper function to train the classifier with the given configuration of the config file."""
    target_id = config["target_id"]
    if target_id is None:
        raise ValueError("Target ID not specified in the config file. Please specify a target ID for the classifier (e.g. 'breed_id').")
    n_epochs = config["n_epochs_cls"]
    lr = config["learning_rate_cls"]
    fit_classification(classifier=classifier,
                       target_id=target_id,
                       n_epochs=n_epochs,
                       lr=lr,
                       train_loader=train_loader,
                       val_loader=val_loader,
                       device=device
                       )
    return classifier
