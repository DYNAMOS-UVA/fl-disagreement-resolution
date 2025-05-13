import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime

from models import create_model
from data_utils import (
    load_client_data,
    preprocess_ncmapss_data,
    create_client_dataloaders,
    load_mnist_client_data,
    create_mnist_dataloaders
)

class FederatedClient:
    def __init__(
        self,
        client_id,
        experiment_type,
        data_dir,
        batch_size=64,
        epochs=5,
        learning_rate=0.001,
        device=None
    ):
        self.client_id = client_id
        self.experiment_type = experiment_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and training parameters based on experiment type
        if experiment_type == "n_cmapss":
            self.seq_len = 50
            self.n_features = 20
            self.input_dim = self.seq_len * self.n_features
            self.hidden_dim = 32
            self.output_dim = 1
            self.model = create_model(
                experiment_type,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim
            ).to(self.device)
        elif experiment_type == "mnist":
            self.model = create_model(experiment_type).to(self.device)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

        # Prepare results directory
        os.makedirs("output/client_results", exist_ok=True)

    def load_data(self, sample_size=1000):
        """Load and preprocess client data"""
        if self.experiment_type == "n_cmapss":
            # Load data for this client
            samples, labels = load_client_data(
                self.client_id,
                self.data_dir,
                sample_size=sample_size
            )

            # Preprocess data
            samples_normalized, _ = preprocess_ncmapss_data(samples)

            # Create dataloaders
            self.train_loader, self.valid_loader = create_client_dataloaders(
                samples_normalized,
                labels,
                batch_size=self.batch_size
            )

            print(f"Client {self.client_id} loaded {len(samples)} samples")
        elif self.experiment_type == "mnist":
            # Load MNIST data for this client
            images, labels = load_mnist_client_data(
                self.client_id,
                train_dir=self.data_dir,
                sample_size=sample_size
            )

            # Create dataloaders
            self.train_loader, self.valid_loader = create_mnist_dataloaders(
                images,
                labels,
                batch_size=self.batch_size
            )

            print(f"Client {self.client_id} loaded {len(images)} MNIST samples")
        else:
            # For other experiments
            raise NotImplementedError(f"{self.experiment_type} data loading not implemented yet")

    def train(self, epochs=None):
        """Train the model on client data"""
        epochs = epochs or self.epochs

        # Set criterion based on experiment type
        if self.experiment_type == "n_cmapss":
            criterion = nn.MSELoss()
        elif self.experiment_type == "mnist":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses = []
        valid_losses = []

        self.model.train()
        print(f"Client {self.client_id} starting training for {epochs} epochs")

        for epoch in range(epochs):
            # Training
            train_loss = 0
            self.model.train()
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Calculate accuracy for MNIST
                if self.experiment_type == "mnist":
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            # Calculate average training loss
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)

            # Calculate training accuracy for MNIST
            train_acc = correct / total if self.experiment_type == "mnist" else None

            # Validation
            valid_loss = 0
            self.model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in self.valid_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    valid_loss += loss.item()

                    # Calculate accuracy for MNIST
                    if self.experiment_type == "mnist":
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()

            # Calculate average validation loss
            valid_loss /= len(self.valid_loader)
            valid_losses.append(valid_loss)

            # Calculate validation accuracy for MNIST
            valid_acc = val_correct / val_total if self.experiment_type == "mnist" else None

            # Print progress
            if self.experiment_type == "mnist":
                print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                      f"Valid Loss: {valid_loss:.6f}, Valid Acc: {valid_acc:.4f}")
            else:
                print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")

        # Save training results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "client_id": self.client_id,
            "experiment_type": self.experiment_type,
            "epochs": epochs,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "timestamp": timestamp
        }

        with open(f"output/client_results/client_{self.client_id}_{timestamp}.json", "w") as f:
            json.dump(results, f)

        print(f"Client {self.client_id} finished training")
        return train_losses, valid_losses

    def get_model_parameters(self):
        """Get model parameters to send to the server"""
        return self.model.get_parameters()

    def set_model_parameters(self, parameters):
        """Update model with parameters from the server"""
        self.model.set_parameters(parameters)
        print(f"Client {self.client_id} updated model with parameters from server")

# Main function for standalone client execution
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--experiment", type=str, default="n_cmapss", choices=["n_cmapss", "mnist"], help="Experiment type")
    parser.add_argument("--data_dir", type=str, help="Data directory (defaults to experiment-specific location)")
    parser.add_argument("--sample_size", type=int, default=1000, help="Sample size per client")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    # Set default data directory based on experiment type if not provided
    if args.data_dir is None:
        if args.experiment == "n_cmapss":
            args.data_dir = "data/n-cmapss/train"
        elif args.experiment == "mnist":
            args.data_dir = "data/mnist/train"

    # Create and run client
    client = FederatedClient(
        client_id=args.client_id,
        experiment_type=args.experiment,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )

    client.load_data(sample_size=args.sample_size)
    client.train()

if __name__ == "__main__":
    main()
