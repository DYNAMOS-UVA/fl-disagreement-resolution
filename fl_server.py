"""Federated learning server implementation."""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob

from models import create_model
import data_module

class FederatedServer:
    """Server-side implementation for federated learning."""

    def __init__(
        self,
        experiment_type,
        test_dir=None,
        test_units=None,
        device=None,
        storage_dir=None
    ):
        """Initialize the federated learning server.

        Args:
            experiment_type: Type of experiment ('n_cmapss' or 'mnist')
            test_dir: Directory containing test data
            test_units: List of unit IDs to use for testing (for N-CMAPSS)
            device: Device to run the model on ('cuda' or 'cpu')
            storage_dir: Directory for storing models and results
        """
        self.experiment_type = experiment_type
        self.test_dir = test_dir
        self.test_units = test_units
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.storage_dir = storage_dir
        self.round = 0
        self.global_model = None
        self.client_models = {}
        self.aggregation_weights = {}
        self.training_history = {
            "rounds": [],
            "global_test_loss": [],
            "global_test_accuracy": []  # For classification tasks like MNIST
        }

        # Create output directories
        if storage_dir:
            self.output_dir = os.path.join(storage_dir, "output", "server")
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        else:
            os.makedirs("output/server_results", exist_ok=True)
            os.makedirs("output/plots", exist_ok=True)
            os.makedirs("output/models", exist_ok=True)

        # Initialize model based on experiment type
        self._init_model()

    def _init_model(self):
        """Initialize global model based on experiment type."""
        if self.experiment_type == "n_cmapss":
            self.seq_len = 50
            self.n_features = 20
            self.input_dim = self.seq_len * self.n_features
            self.hidden_dim = 32
            self.output_dim = 1
            self.global_model = create_model(
                self.experiment_type,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim
            ).to(self.device)
        elif self.experiment_type == "mnist":
            self.global_model = create_model(self.experiment_type).to(self.device)
        else:
            raise ValueError(f"Unsupported experiment type: {self.experiment_type}")

        print(f"Initialized global {self.experiment_type} model")

    def load_test_data(self, sample_size=500):
        """Load test data for model evaluation.

        Args:
            sample_size: Maximum number of samples to load per test unit
        """
        if self.experiment_type == "n_cmapss":
            if not self.test_dir or not self.test_units:
                raise ValueError("Test directory and test units must be provided for N-CMAPSS")

            # Load test data
            test_samples, test_labels = data_module.load_ncmapss_test_data(
                self.test_dir,
                self.test_units,
                sample_size=sample_size
            )

            # Preprocess test data
            _, test_normalized, _ = data_module.preprocess_ncmapss_data(test_samples, test_samples)

            # Create test dataloader
            self.test_loader = data_module.create_ncmapss_test_dataloader(
                test_normalized,
                test_labels,
                batch_size=64
            )

            print(f"Loaded test data with {len(test_samples)} samples")
        elif self.experiment_type == "mnist":
            if not self.test_dir:
                raise ValueError("Test directory must be provided for MNIST")

            # Load MNIST test data
            test_images, test_labels = data_module.load_mnist_test_data(
                test_dir=self.test_dir
            )

            # Create test dataloader - no preprocessing needed as it's done during download
            self.test_loader = data_module.create_mnist_test_dataloader(
                test_images,
                test_labels,
                batch_size=64
            )

            print(f"Loaded MNIST test data with {len(test_images)} samples")
        else:
            # To be implemented for other experiments
            raise NotImplementedError(f"{self.experiment_type} test data loading not implemented yet")

    def get_model_parameters(self):
        """Get current global model parameters to send to clients.

        Returns:
            list: List of model parameter tensors
        """
        return self.global_model.get_parameters()

    def save_model(self, model_dir):
        """Save global model to a directory.

        Args:
            model_dir: Directory to save the model to
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save model state dict
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.global_model.state_dict(), model_path)

        # Save model metadata
        metadata = {
            "experiment_type": self.experiment_type,
            "round": self.round,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved global model to {model_dir}")

    def load_model(self, model_dir):
        """Load global model from a directory.

        Args:
            model_dir: Directory containing the model
        """
        model_path = os.path.join(model_dir, "model.pt")
        metadata_path = os.path.join(model_dir, "metadata.json")

        # Load model state dict
        self.global_model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load metadata if available
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if "round" in metadata:
                    # Keep track of the round but don't overwrite current round
                    print(f"Loaded model from round {metadata['round']}")

        print(f"Loaded global model from {model_dir}")

    def aggregate_models(self, client_parameters, aggregation_weights=None):
        """Aggregate model parameters from clients using weighted average.

        Args:
            client_parameters: Dictionary mapping client IDs to model parameters
            aggregation_weights: Optional dictionary mapping client IDs to weights

        Returns:
            list: List of aggregated model parameter tensors
        """
        self.round += 1
        print(f"Starting aggregation round {self.round}")

        # If no weights provided, use equal weighting
        if aggregation_weights is None:
            n_clients = len(client_parameters)
            aggregation_weights = {client_id: 1.0 / n_clients for client_id in client_parameters.keys()}

        # Store client models and weights for this round
        self.client_models = client_parameters
        self.aggregation_weights = aggregation_weights

        # Initialize new global parameters with zeros
        global_parameters = [torch.zeros_like(param) for param in self.global_model.parameters()]

        # Weighted average of client parameters
        for client_id, parameters in client_parameters.items():
            weight = aggregation_weights[client_id]
            for i, param in enumerate(parameters):
                global_parameters[i] += param * weight

        # Update global model
        self.global_model.set_parameters(global_parameters)
        print(f"Updated global model with parameters from {len(client_parameters)} clients")

        return global_parameters

    def aggregate_models_from_files(self, clients_dir, aggregation_weights=None):
        """Aggregate models from client files.

        Args:
            clients_dir: Directory containing client model directories
            aggregation_weights: Optional dictionary mapping client IDs to weights

        Returns:
            list: List of aggregated model parameter tensors
        """
        self.round += 1
        print(f"Starting file-based aggregation for round {self.round}")

        # Find all client directories
        client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
        client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]

        print(f"Found {len(client_dirs)} clients: {client_ids}")

        # If no weights provided, use equal weighting
        if aggregation_weights is None:
            n_clients = len(client_dirs)
            aggregation_weights = {client_id: 1.0 / n_clients for client_id in client_ids}

        # Initialize temporary model for loading client models
        temp_model = create_model(
            self.experiment_type,
            input_dim=self.input_dim if self.experiment_type == "n_cmapss" else None,
            hidden_dim=self.hidden_dim if self.experiment_type == "n_cmapss" else None,
            output_dim=self.output_dim if self.experiment_type == "n_cmapss" else None
        ).to(self.device)

        # Initialize new global parameters with zeros
        global_parameters = [torch.zeros_like(param) for param in self.global_model.parameters()]

        # Load each client model and aggregate parameters
        for client_dir, client_id in zip(client_dirs, client_ids):
            # Load client model
            model_path = os.path.join(client_dir, "model.pt")
            temp_model.load_state_dict(torch.load(model_path, map_location=self.device))

            # Get client parameters
            client_parameters = temp_model.get_parameters()

            # Get client weight
            weight = aggregation_weights.get(client_id, 1.0 / len(client_dirs))

            # Add weighted parameters to global parameters
            for i, param in enumerate(client_parameters):
                global_parameters[i] += param * weight

        # Update global model with aggregated parameters
        self.global_model.set_parameters(global_parameters)
        print(f"Updated global model with parameters from {len(client_dirs)} clients")

        return global_parameters

    def evaluate_model(self):
        """Evaluate global model on test data.

        Returns:
            tuple: (test_loss, accuracy) where accuracy is None for regression tasks
        """
        self.global_model.eval()

        # Set criterion based on experiment type
        if self.experiment_type == "n_cmapss":
            criterion = nn.MSELoss()
        elif self.experiment_type == "mnist":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")

        test_loss = 0
        predictions = []
        actual = []
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                test_loss += loss.item()

                # For regression (N-CMAPSS)
                if self.experiment_type == "n_cmapss":
                    predictions.extend(output.cpu().numpy())
                    actual.extend(target.cpu().numpy())
                # For classification (MNIST)
                elif self.experiment_type == "mnist":
                    _, predicted = torch.max(output.data, 1)
                    predictions.extend(predicted.cpu().numpy())
                    actual.extend(target.cpu().numpy())
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

        # Calculate average test loss
        test_loss /= len(self.test_loader)

        # For RUL prediction, calculate RMSE
        if self.experiment_type == "n_cmapss":
            test_loss = np.sqrt(test_loss)
            accuracy = None
        # For classification, calculate accuracy
        elif self.experiment_type == "mnist":
            accuracy = correct / total
            print(f"Round {self.round} - Global model test accuracy: {accuracy:.4f}")

        # Store history
        self.training_history["rounds"].append(self.round)
        self.training_history["global_test_loss"].append(test_loss)
        if accuracy is not None:
            self.training_history["global_test_accuracy"].append(accuracy)

        print(f"Round {self.round} - Global model test loss: {test_loss:.6f}")

        # Plot and save results
        self._save_results(predictions, actual)

        return test_loss, accuracy

    def _save_results(self, predictions, actual):
        """Save evaluation results and plots.

        Args:
            predictions: List of model predictions
            actual: List of actual values
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine output paths based on storage_dir
        if self.storage_dir:
            history_path = os.path.join(self.output_dir, f"training_history_round_{self.round}.json")
            loss_plot_path = os.path.join(self.output_dir, "plots", f"global_model_loss_round_{self.round}.png")

            if self.experiment_type == "n_cmapss":
                pred_plot_path = os.path.join(self.output_dir, "plots", f"rul_prediction_round_{self.round}.png")
            else:
                cm_plot_path = os.path.join(self.output_dir, "plots", f"mnist_confusion_matrix_round_{self.round}.png")
                acc_plot_path = os.path.join(self.output_dir, "plots", f"global_model_accuracy_round_{self.round}.png")
        else:
            history_path = f"output/server_results/training_history_round_{self.round}_{timestamp}.json"
            loss_plot_path = f"output/plots/global_model_loss_round_{self.round}_{timestamp}.png"

            if self.experiment_type == "n_cmapss":
                pred_plot_path = f"output/plots/rul_prediction_round_{self.round}_{timestamp}.png"
            else:
                cm_plot_path = f"output/plots/mnist_confusion_matrix_round_{self.round}_{timestamp}.png"
                acc_plot_path = f"output/plots/global_model_accuracy_round_{self.round}_{timestamp}.png"

        # Save training history
        with open(history_path, "w") as f:
            json.dump(self.training_history, f)

        # Plot and save loss history
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history["rounds"], self.training_history["global_test_loss"], marker='o')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Test Loss')
        plt.title(f'Global Model Performance ({self.experiment_type})')
        plt.grid(True)
        plt.savefig(loss_plot_path)
        plt.close()

        # For RUL prediction, plot predictions vs actual
        if self.experiment_type == "n_cmapss":
            predictions = np.array(predictions)
            actual = np.array(actual)

            plt.figure(figsize=(10, 6))
            plt.scatter(actual, predictions, alpha=0.5)
            plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r')
            plt.xlabel('Actual RUL')
            plt.ylabel('Predicted RUL')
            plt.title(f'RUL Prediction (RMSE: {self.training_history["global_test_loss"][-1]:.4f})')
            plt.savefig(pred_plot_path)
            plt.close()

        # For MNIST, plot confusion matrix and accuracy
        elif self.experiment_type == "mnist":
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(actual, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix (Accuracy: {self.training_history["global_test_accuracy"][-1]:.4f})')
            plt.savefig(cm_plot_path)
            plt.close()

            # Plot accuracy history if we have at least 2 rounds
            if len(self.training_history["global_test_accuracy"]) >= 2:
                plt.figure(figsize=(10, 6))
                plt.plot(self.training_history["rounds"], self.training_history["global_test_accuracy"], marker='o')
                plt.xlabel('Federated Learning Round')
                plt.ylabel('Test Accuracy')
                plt.title('Global Model Accuracy (MNIST)')
                plt.grid(True)
                plt.savefig(acc_plot_path)
                plt.close()

        # Save model
        if self.storage_dir:
            model_path = os.path.join(self.storage_dir, f"output/models/round_{self.round}_model.pt")
        else:
            model_path = f'output/models/{self.experiment_type}_global_model_round_{self.round}_{timestamp}.pth'

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.global_model.state_dict(), model_path)

        print(f"Saved results for round {self.round}")


def main():
    """Run the server as a standalone application."""
    import argparse

    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--experiment", type=str, default="n_cmapss", choices=["n_cmapss", "mnist"], help="Experiment type")
    parser.add_argument("--test_dir", type=str, help="Test data directory (defaults to experiment-specific location)")
    parser.add_argument("--test_units", type=int, nargs="+", default=[11, 14, 15], help="Test units (for N-CMAPSS)")
    parser.add_argument("--sample_size", type=int, default=500, help="Sample size per test unit (for N-CMAPSS)")
    parser.add_argument("--storage_dir", type=str, help="Storage directory for models and results")

    args = parser.parse_args()

    # Set default test directory based on experiment type if not provided
    if args.test_dir is None:
        if args.experiment == "n_cmapss":
            args.test_dir = "data/n-cmapss/test"
        elif args.experiment == "mnist":
            args.test_dir = "data/mnist/test"

    # Create server
    server = FederatedServer(
        experiment_type=args.experiment,
        test_dir=args.test_dir,
        test_units=args.test_units if args.experiment == "n_cmapss" else None,
        storage_dir=args.storage_dir
    )

    # Load test data
    if args.experiment == "n_cmapss":
        server.load_test_data(sample_size=args.sample_size)
    else:
        server.load_test_data()

    # Note: This doesn't actually run aggregation since it needs client models
    # This is just for testing the server initialization
    print("Server initialized successfully. In practice, it would be called by the orchestrator.")


if __name__ == "__main__":
    main()
