"""Main federated learning server implementation."""

import os
import torch
import torch.nn as nn
import json
from datetime import datetime
import glob

from fl_module import create_model
import fl_module
from fl_server.evaluation import evaluate_model
from fl_server.aggregation import aggregate_models_from_files
from fl_server.utils import make_json_serializable

class FederatedServer:
    """Server-side implementation for federated learning."""

    def __init__(
        self,
        experiment_type,
        test_dir=None,
        test_units=None,
        device=None,
        results_dir=None
    ):
        """Initialize the federated learning server.

        Args:
            experiment_type: Type of experiment ('n_cmapss' or 'mnist')
            test_dir: Directory containing test data
            test_units: List of unit IDs to use for testing (for N-CMAPSS)
            device: Device to run the model on ('cuda' or 'cpu')
            results_dir: Directory for storing models and results
        """
        self.experiment_type = experiment_type
        self.test_dir = test_dir
        self.test_units = test_units
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = results_dir
        self.round = 0
        self.global_model = None
        self.client_models = {}
        self.aggregation_weights = {}
        self.training_history = {
            "rounds": [],
            "global_test_loss": [],
            "global_test_accuracy": []  # For classification tasks like MNIST
        }

        # Experiment metadata (to be initialized later)
        self.fl_rounds = None
        self.client_ids = None
        self.iid = None

        # Results tracking
        self.results = {
            "experiment_type": experiment_type,
            "rounds": []
        }

        # Create output directories
        if results_dir:
            self.output_dir = os.path.join(results_dir, "output", "server")
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
            # Create global_models directory
            os.makedirs(os.path.join(results_dir, "output", "global_models"), exist_ok=True)
        else:
            os.makedirs("output/server_results", exist_ok=True)
            os.makedirs("output/plots", exist_ok=True)
            os.makedirs("output/global_models", exist_ok=True)

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
            test_samples, test_labels = fl_module.load_ncmapss_test_data(
                self.test_dir,
                self.test_units,
                sample_size=sample_size
            )

            # Preprocess test data
            _, test_normalized, _ = fl_module.preprocess_ncmapss_data(test_samples, test_samples)

            # Create test dataloader
            self.test_loader = fl_module.create_ncmapss_test_dataloader(
                test_normalized,
                test_labels,
                batch_size=64
            )

            print(f"Loaded test data with {len(test_samples)} samples")
        elif self.experiment_type == "mnist":
            if not self.test_dir:
                raise ValueError("Test directory must be provided for MNIST")

            # Load MNIST test data
            test_images, test_labels = fl_module.load_mnist_test_data(
                test_dir=self.test_dir
            )

            # Create test dataloader - no preprocessing needed as it's done during download
            self.test_loader = fl_module.create_mnist_test_dataloader(
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

    def create_model_dirs(self, round_num=None, structure=None):
        """Create necessary directories for models.

        Args:
            round_num: Round number (None for initial model directory)
            structure: Dictionary with directory structure information
        """
        if not self.results_dir or not structure:
            return None

        # Create initial model directory
        if round_num is None:
            initial_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
            os.makedirs(initial_model_dir, exist_ok=True)
            return initial_model_dir

        # Create round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )
        os.makedirs(round_dir, exist_ok=True)

        # Create directory for the global model for training
        training_model_dir = os.path.join(round_dir, structure["global_model"])
        os.makedirs(training_model_dir, exist_ok=True)

        # Create directory for the aggregated model
        aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
        os.makedirs(aggregated_model_dir, exist_ok=True)

        return {
            "round_dir": round_dir,
            "training_model_dir": training_model_dir,
            "aggregated_model_dir": aggregated_model_dir
        }

    def get_model_dir_paths(self, round_num=None, aggregated=False, structure=None):
        """Get paths for model directories.

        Args:
            round_num: Round number (None for initial model)
            aggregated: Whether to get the aggregated model directory
            structure: Dictionary with directory structure information

        Returns:
            str: Path to the model directory
        """
        if not self.results_dir or not structure:
            return None

        # Initial global model
        if round_num is None:
            dir_path = os.path.join(self.results_dir, structure["global_model_initial"])
            os.makedirs(dir_path, exist_ok=True)
            return dir_path

        # Round-specific global model
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        if aggregated:
            dir_path = os.path.join(round_dir, structure["global_model_aggregated"])
        else:
            dir_path = os.path.join(round_dir, structure["global_model"])

        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def init_experiment(self, fl_rounds, client_ids, iid=False):
        """Initialize experiment metadata.

        Args:
            fl_rounds: Number of federated learning rounds
            client_ids: List of client IDs
            iid: Whether the data distribution is IID (for MNIST)
        """
        self.fl_rounds = fl_rounds
        self.client_ids = client_ids
        self.iid = iid

        # Update results with experiment metadata
        self.results["fl_rounds"] = fl_rounds
        self.results["client_ids"] = client_ids
        self.results["iid"] = iid if self.experiment_type == "mnist" else None

        # Save initial results
        self._save_experiment_results()

    def _save_experiment_results(self):
        """Save experiment results to a JSON file."""
        # Check if we have a results directory
        if not self.results_dir:
            return

        # Create the output directory if it doesn't exist
        output_dir = os.path.join(self.results_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Convert NumPy types to Python native types
        serializable_results = make_json_serializable(self.results)

        # Save the results
        results_path = os.path.join(output_dir, "fl_results.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Saved experiment results to {results_path}")

    def initialize_model(self, round_num=0):
        """Initialize and save the initial model.

        Args:
            round_num: Usually 0 for the initial model
        """
        if not self.results_dir:
            return

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Create the initial model directory
        initial_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
        os.makedirs(initial_model_dir, exist_ok=True)

        # Save the initial model
        self.save_model(initial_model_dir)
        print(f"Initialized and saved the initial global model")

    def prepare_training_model(self, round_num, use_initial=False):
        """Prepare the global model for a specific round.

        Args:
            round_num: The current round number
            use_initial: Whether to use the initial model (for round 1)

        Returns:
            str: Path to the prepared model directory
        """
        if not self.results_dir:
            return

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Create round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )
        os.makedirs(round_dir, exist_ok=True)

        # Create directory for the global model for training
        training_model_dir = os.path.join(round_dir, structure["global_model"])
        os.makedirs(training_model_dir, exist_ok=True)

        # Create directory for the aggregated model
        aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
        os.makedirs(aggregated_model_dir, exist_ok=True)

        # Load the appropriate source model
        if use_initial:
            # Use the initial model for round 1
            source_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
        else:
            # Use the previous round's aggregated model
            prev_round_dir = os.path.join(
                self.results_dir,
                structure["round_template"].format(round=round_num-1)
            )
            source_model_dir = os.path.join(prev_round_dir, structure["global_model_aggregated"])

        # Load the source model
        self.load_model(source_model_dir)

        # Save it as the training model for this round
        self.save_model(training_model_dir)

        return training_model_dir

    def aggregate_client_models(self, round_num):
        """Aggregate client models for a specific round.

        Args:
            round_num: The current round number

        Returns:
            bool: Whether the aggregation was successful
        """
        if not self.results_dir:
            return False

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Get the round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        # Get the clients directory
        clients_dir = os.path.join(round_dir, structure["clients_dir"])

        # Get the aggregated model directory
        aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
        os.makedirs(aggregated_model_dir, exist_ok=True)

        # Aggregate client models
        aggregate_models_from_files(self, clients_dir)

        # Save the aggregated model
        self.save_model(aggregated_model_dir)

        return True

    def _get_structure_config(self):
        """Get the directory structure configuration.

        Returns:
            dict: Directory structure configuration
        """
        # Default structure configuration
        default_structure = {
            "global_model_initial": "global_model_initial",
            "round_template": "round_{round}",
            "clients_dir": "clients",
            "global_model": "global_model_for_training",
            "global_model_aggregated": "global_model_aggregated",
            "client_prefix": "client_"
        }

        # Try to load from configuration file
        config_path = os.path.join(os.path.dirname(self.results_dir), "mock_etcd/configuration.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "results" in config and "structure" in config["results"]:
                        return config["results"]["structure"]
        except Exception as e:
            print(f"Error loading configuration: {e}")

        return default_structure

    def evaluate_model(self, fl_round=None, client_results=None):
        """Evaluate global model on test data.

        Args:
            fl_round: Current federated learning round (if None, considered as initial round 0)
            client_results: DEPRECATED - Dictionary of client training results (not used anymore)

        Returns:
            tuple: (test_loss, accuracy) where accuracy is None for regression tasks
        """
        # Delegate to the evaluate_model function in evaluation.py
        return evaluate_model(self, fl_round, client_results)
