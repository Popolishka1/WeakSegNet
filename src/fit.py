import torch

def fit_adam(model, loss_fn, n_epochs, lr, trainloader, device="cpu"):

    print(f"\n----Fitting model with {n_epochs} epochs")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1 + n_epochs):
        model.train()
        n_batches = len(trainloader)
        batch_loss = 0.0
        for images, masks in trainloader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()

        avg_batch_loss = batch_loss / n_batches
        print(f"[Epoch {epoch}/{n_epochs}] - Total loss: {loss.item():.4f} - Batch loss: {avg_batch_loss.item():.4f}")
    print("\nTraining done")
