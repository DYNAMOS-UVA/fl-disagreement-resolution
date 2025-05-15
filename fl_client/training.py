"""Training functionality for federated learning client."""

import torch
import torch.nn as nn
import torch.optim as optim

def train_model(client, epochs):
    """Train the model on client data.

    Args:
        client: FederatedClient instance
        epochs: Number of epochs to train

    Returns:
        dict: Dictionary containing training results
    """
    model = client.model
    train_loader = client.train_loader
    valid_loader = client.valid_loader
    device = client.device
    learning_rate = client.learning_rate
    client_id = client.client_id
    experiment_type = client.experiment_type

    # Set criterion based on experiment type
    if experiment_type == "n_cmapss":
        criterion = nn.MSELoss()
    elif experiment_type == "mnist":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    model.train()
    print(f"Client {client_id} starting training for {epochs} epochs")

    for epoch in range(epochs):
        # Training
        train_loss = 0
        model.train()
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate accuracy for MNIST
            if experiment_type == "mnist":
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Calculate training accuracy for MNIST
        train_acc = correct / total if experiment_type == "mnist" else None
        if train_acc is not None:
            train_accuracies.append(train_acc)

        # Validation
        valid_loss = 0
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()

                # Calculate accuracy for MNIST
                if experiment_type == "mnist":
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()

        # Calculate average validation loss
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        # Calculate validation accuracy for MNIST
        valid_acc = val_correct / val_total if experiment_type == "mnist" else None
        if valid_acc is not None:
            valid_accuracies.append(valid_acc)

        # Print progress
        if experiment_type == "mnist":
            print(f"Client {client_id} - Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                  f"Valid Loss: {valid_loss:.6f}, Valid Acc: {valid_acc:.4f}")
        else:
            print(f"Client {client_id} - Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")

    # Create results dictionary
    training_results = {
        "client_id": client_id,
        "experiment_type": experiment_type,
        "epochs": epochs,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "final_train_loss": train_losses[-1],
        "final_valid_loss": valid_losses[-1],
    }

    # Add accuracy metrics for classification tasks
    if experiment_type == "mnist":
        training_results.update({
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "final_train_accuracy": train_accuracies[-1],
            "final_valid_accuracy": valid_accuracies[-1]
        })

    print(f"Client {client_id} finished training")
    return training_results
