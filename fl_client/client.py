"""Main federated learning client implementation."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime

from models import create_model
import data_module
from fl_client.utils import save_training_results
from fl_client.training import train_model

class FederatedClient:
    """Client-side implementation for federated learning."""

    def __init__(
        self,
        client_id,
        experiment_type,
        data_dir,
        batch_size=64,
        epochs=5,
        learning_rate=0.001,
        device=None,
        storage_dir=None
    ):
        """Initialize the federated learning client.

        Args:
            client_id: Client ID
            experiment_type: Type of experiment ('n_cmapss' or 'mnist')
            data_dir: Directory containing client data
            batch_size: Batch size for training
            epochs: Number of local training epochs
            learning_rate: Learning rate for optimization
            device: Device to run the model on ('cuda' or 'cpu')
            storage_dir: Directory for storing models and results
        """
        self.client_id = client_id
        self.experiment_type = experiment_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.storage_dir = storage_dir

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

        # Create results directory
        if storage_dir:
            self.output_dir = os.path.join(storage_dir, "output", "clients", f"client_{client_id}")
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            os.makedirs("output/client_results", exist_ok=True)

    def create_model_dir(self, round_num, structure=None):
        """Create client model directory for a specific round.

        Args:
            round_num: The round number
            structure: Dictionary with directory structure information

        Returns:
            str: Path to the client model directory
        """
        if not self.storage_dir or not structure:
            return None

        # Create the client directory for this round
        round_dir = os.path.join(
            self.storage_dir,
            structure["round_template"].format(round=round_num)
        )

        clients_dir = os.path.join(round_dir, structure["clients_dir"])
        os.makedirs(clients_dir, exist_ok=True)

        client_dir = os.path.join(clients_dir, f"{structure['client_prefix']}{self.client_id}")
        os.makedirs(client_dir, exist_ok=True)

        return client_dir

    def load_data(self, sample_size=1000):
        """Load and preprocess client data.

        Args:
            sample_size: Maximum number of samples to load
        """
        if self.experiment_type == "n_cmapss":
            # Load data for this client
            samples, labels = data_module.load_ncmapss_client_data(
                self.client_id,
                self.data_dir,
                sample_size=sample_size
            )

            # Preprocess data
            samples_normalized, _ = data_module.preprocess_ncmapss_data(samples)

            # Create dataloaders
            self.train_loader, self.valid_loader = data_module.create_ncmapss_client_dataloaders(
                samples_normalized,
                labels,
                batch_size=self.batch_size
            )

            print(f"Client {self.client_id} loaded {len(samples)} samples")
        elif self.experiment_type == "mnist":
            # Load MNIST data for this client
            images, labels = data_module.load_mnist_client_data(
                self.client_id,
                train_dir=self.data_dir,
                sample_size=sample_size
            )

            # Create dataloaders
            self.train_loader, self.valid_loader = data_module.create_mnist_client_dataloaders(
                images,
                labels,
                batch_size=self.batch_size
            )

            print(f"Client {self.client_id} loaded {len(images)} MNIST samples")
        else:
            # For other experiments
            raise NotImplementedError(f"{self.experiment_type} data loading not implemented yet")

    def save_model(self, model_dir):
        """Save model to a directory.

        Args:
            model_dir: Directory to save the model to
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save model state dict
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save model metadata
        metadata = {
            "client_id": self.client_id,
            "experiment_type": self.experiment_type,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Client {self.client_id} saved model to {model_dir}")

    def load_model(self, model_dir):
        """Load model from a directory.

        Args:
            model_dir: Directory containing the model
        """
        model_path = os.path.join(model_dir, "model.pt")

        # Load model state dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Client {self.client_id} loaded model from {model_dir}")

    def train(self, epochs=None):
        """Train the model on client data.

        Args:
            epochs: Number of epochs to train (defaults to self.epochs)

        Returns:
            dict: Dictionary containing training results
        """
        epochs = epochs or self.epochs
        training_results = train_model(self, epochs)
        save_training_results(self, training_results)
        return training_results

    def get_model_parameters(self):
        """Get model parameters to send to the server.

        Returns:
            list: List of model parameter tensors
        """
        return self.model.get_parameters()

    def set_model_parameters(self, parameters):
        """Update model with parameters from the server.

        Args:
            parameters: List of model parameter tensors
        """
        self.model.set_parameters(parameters)
        print(f"Client {self.client_id} updated model with parameters from server")

    def load_round_model(self, round_num):
        """Load the global model for a specific round.

        Args:
            round_num: The current round number

        Returns:
            bool: Whether the model was successfully loaded
        """
        if not self.storage_dir:
            return False

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Get the round directory
        round_dir = os.path.join(
            self.storage_dir,
            structure["round_template"].format(round=round_num)
        )

        # Get the global model directory
        global_model_dir = os.path.join(round_dir, structure["global_model"])

        # Load the model
        self.load_model(global_model_dir)
        return True

    def save_round_model(self, round_num):
        """Save the trained model for a specific round.

        Args:
            round_num: The current round number

        Returns:
            str: Path to the saved model directory
        """
        if not self.storage_dir:
            return None

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Create the client directory for this round
        round_dir = os.path.join(
            self.storage_dir,
            structure["round_template"].format(round=round_num)
        )

        clients_dir = os.path.join(round_dir, structure["clients_dir"])
        os.makedirs(clients_dir, exist_ok=True)

        client_dir = os.path.join(clients_dir, f"{structure['client_prefix']}{self.client_id}")
        os.makedirs(client_dir, exist_ok=True)

        # Save the model
        self.save_model(client_dir)
        return client_dir

    def _get_structure_config(self):
        """Get the directory structure configuration.

        Returns:
            dict: Directory structure configuration
        """
        # Default structure configuration
        default_structure = {
            "round_template": "round_{round}",
            "clients_dir": "clients",
            "global_model": "global_model_for_training",
            "client_prefix": "client_"
        }

        # Try to load from configuration file
        config_path = os.path.join(os.path.dirname(self.storage_dir), "mock_etcd/configuration.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "storage" in config and "structure" in config["storage"]:
                        return config["storage"]["structure"]
        except Exception as e:
            print(f"Error loading configuration: {e}")

        return default_structure
