import torch


def fit_adam(model, loss_fn, n_epochs, lr, train_loader, val_loader, device="cpu"):

    print(f"\n----Fitting model with {n_epochs} epochs")
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
            loss = loss_fn(outputs, target_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
        # TODO: might be a good idea to include early stop (with dice/pix acc or val loss)
        avg_train_loss = train_loss / n_batches_train

        # ------- VALIDATION -------
        model.eval()
        n_batches_val = len(val_loader)
        val_loss = 0.0

        with torch.no_grad():
            for images, true_masks, _ in val_loader:
                images = images.to(device)
                true_masks = true_masks.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, true_masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / n_batches_val

        print(f"[Epoch {epoch}/{n_epochs}] Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss) # adjust the lr
    print("[Training done]")
