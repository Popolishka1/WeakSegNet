import torch
import torch.nn as nn


def fit_segmentation(model, n_epochs, lr, train_loader, val_loader, device="cpu"):
    print(f"\n----Fitting segmentation model with {n_epochs} epochs")

    criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) # lr dynamic adjustment 

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
        avg_train_loss = train_loss / n_batches_train # TODO: might be a good idea to include early stop (with dice/pix acc or val loss)

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

# TODO: this will be useful for CAM classification
def fit_classification(classifier, n_epochs, lr, train_loader, val_loader, device="cpu"):
    print(f"\n----Fitting classifier with {n_epochs} epochs")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) # lr dynamic adjustment 

    for epoch in range(1, n_epochs + 1):

        # ------- TRAINING -------
        classifier.train()
        n_batches_train = len(train_loader)
        train_loss = 0.0

        for images, _, info in train_loader:
            images = images.to(device)
            labels = info["breed_id"].clone().detach().to(device)

            outputs = classifier(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() 
        avg_train_loss = train_loss / n_batches_train # TODO: might be a good idea to include early stop (with dice/pix acc or val loss)

        # ------- VALIDATION -------
        classifier.eval()
        n_batches_val = len(val_loader)
        val_loss = 0.0

        with torch.no_grad():
            for images, _, info in val_loader:
                images = images.to(device)
                labels = info["breed_id"].clone().detach().to(device)

                outputs = classifier(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
        avg_val_loss = val_loss / n_batches_val

        print(f"[Classifier] Epoch {epoch}/{n_epochs} -> Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss) # adjust the lr
    print("[Classifier training done]")
