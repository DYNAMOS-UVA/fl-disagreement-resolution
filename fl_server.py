"""Federated learning server implementation."""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
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
        if storage_dir:
            self.output_dir = os.path.join(storage_dir, "output", "server")
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
            # Create global_models directory
            os.makedirs(os.path.join(storage_dir, "output", "global_models"), exist_ok=True)
        else:
            os.makedirs("output/server_results", exist_ok=True)
            os.makedirs("output/plots", exist_ok=True)
            os.makedirs("output/global_models", exist_ok=True)

        # Initialize model based on experiment type
        self._init_model()

    def create_model_dirs(self, round_num=None, structure=None):
        """Create necessary directories for models.

        Args:
            round_num: Round number (None for initial model directory)
            structure: Dictionary with directory structure information
        """
        if not self.storage_dir or not structure:
            return None

        # Create initial model directory
        if round_num is None:
            initial_model_dir = os.path.join(self.storage_dir, structure["global_model_initial"])
            os.makedirs(initial_model_dir, exist_ok=True)
            return initial_model_dir

        # Create round directory
        round_dir = os.path.join(
            self.storage_dir,
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
        if not self.storage_dir or not structure:
            return None

        # Initial global model
        if round_num is None:
            dir_path = os.path.join(self.storage_dir, structure["global_model_initial"])
            os.makedirs(dir_path, exist_ok=True)
            return dir_path

        # Round-specific global model
        round_dir = os.path.join(
            self.storage_dir,
            structure["round_template"].format(round=round_num)
        )

        if aggregated:
            dir_path = os.path.join(round_dir, structure["global_model_aggregated"])
        else:
            dir_path = os.path.join(round_dir, structure["global_model"])

        os.makedirs(dir_path, exist_ok=True)
        return dir_path

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

    def evaluate_model(self, fl_round=None, client_results=None):
        """Evaluate global model on test data.

        Args:
            fl_round: Current federated learning round (if None, considered as initial round 0)
            client_results: DEPRECATED - Dictionary of client training results (not used anymore)

        Returns:
            tuple: (test_loss, accuracy) where accuracy is None for regression tasks
        """
        self.global_model.eval()

        # If round is not provided, use the current round counter
        if fl_round is None:
            fl_round = self.round
        else:
            # Update the internal round counter if a specific round is provided
            self.round = fl_round

        # Read client results from filesystem if not round 0
        if fl_round > 0 and not client_results:
            client_results = self._read_client_results_from_files(fl_round)

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

        # For RUL prediction, calculate RMSE and additional metrics
        if self.experiment_type == "n_cmapss":
            rmse = np.sqrt(test_loss)
            test_loss = rmse  # Keep RMSE as the primary test loss metric

            # Convert to numpy arrays for calculation
            predictions = np.array(predictions)
            actual = np.array(actual)

            # Calculate Mean Absolute Error (MAE)
            mae = np.mean(np.abs(predictions - actual))

            # Calculate R² (coefficient of determination)
            mean_actual = np.mean(actual)
            ss_total = np.sum((actual - mean_actual) ** 2)
            ss_residual = np.sum((actual - predictions) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            # Calculate % of predictions within ±10 cycles (a more intuitive metric)
            within_10_cycles = np.mean(np.abs(predictions - actual) <= 10.0) * 100
            within_20_cycles = np.mean(np.abs(predictions - actual) <= 20.0) * 100

            # Print all metrics
            print(f"Round {self.round} - RUL Prediction Metrics:")
            print(f"  RMSE: {rmse:.2f} cycles")
            print(f"  MAE: {mae:.2f} cycles")
            print(f"  R²: {r_squared:.4f}")
            print(f"  Within ±10 cycles: {within_10_cycles:.2f}%")
            print(f"  Within ±20 cycles: {within_20_cycles:.2f}%")

            # Store additional metrics in training history
            if "rul_mae" not in self.training_history:
                self.training_history["rul_mae"] = []
            if "rul_r_squared" not in self.training_history:
                self.training_history["rul_r_squared"] = []
            if "rul_within_10" not in self.training_history:
                self.training_history["rul_within_10"] = []
            if "rul_within_20" not in self.training_history:
                self.training_history["rul_within_20"] = []

            self.training_history["rul_mae"].append(mae)
            self.training_history["rul_r_squared"].append(r_squared)
            self.training_history["rul_within_10"].append(within_10_cycles)
            self.training_history["rul_within_20"].append(within_20_cycles)

            # Update results dictionary
            round_results = {
                "round": fl_round,
                "test_loss": test_loss,
                "mae": mae,
                "r_squared": r_squared,
                "within_10_cycles": within_10_cycles,
                "within_20_cycles": within_20_cycles
            }

            # Add client results if provided
            if client_results:
                round_results["client_results"] = client_results

            # Add to results history
            self.results["rounds"].append(round_results)

            accuracy = None

        # For classification, calculate accuracy and other metrics
        elif self.experiment_type == "mnist":
            accuracy = correct / total

            # Calculate precision, recall, and F1 score for each class
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predictions, average=None, zero_division=0
            )

            # Calculate mean metrics (weighted by support)
            mean_precision, mean_recall, mean_f1, _ = precision_recall_fscore_support(
                actual, predictions, average='weighted', zero_division=0
            )

            # Calculate per-class accuracy
            class_labels = np.unique(actual)
            per_class_accuracy = []
            for c in class_labels:
                # Mask for this class
                mask = np.array(actual) == c
                # Accuracy for this class
                class_acc = np.mean(np.array(predictions)[mask] == c) if np.sum(mask) > 0 else 0
                per_class_accuracy.append(class_acc)

            # Print detailed metrics
            print(f"Round {self.round} - MNIST Classification Metrics:")
            print(f"  Overall Accuracy: {accuracy:.4f}")
            print(f"  Mean Precision: {mean_precision:.4f}")
            print(f"  Mean Recall: {mean_recall:.4f}")
            print(f"  Mean F1 Score: {mean_f1:.4f}")

            # Store additional metrics in training history
            if "mnist_precision" not in self.training_history:
                self.training_history["mnist_precision"] = []
            if "mnist_recall" not in self.training_history:
                self.training_history["mnist_recall"] = []
            if "mnist_f1" not in self.training_history:
                self.training_history["mnist_f1"] = []
            if "mnist_per_class_accuracy" not in self.training_history:
                self.training_history["mnist_per_class_accuracy"] = []

            self.training_history["mnist_precision"].append(mean_precision)
            self.training_history["mnist_recall"].append(mean_recall)
            self.training_history["mnist_f1"].append(mean_f1)
            self.training_history["mnist_per_class_accuracy"].append(per_class_accuracy)

            print(f"Round {self.round} - Global model test accuracy: {accuracy:.4f}")

            # Update results dictionary
            round_results = {
                "round": fl_round,
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "mean_precision": mean_precision if 'mean_precision' in locals() else None,
                "mean_recall": mean_recall if 'mean_recall' in locals() else None,
                "mean_f1": mean_f1 if 'mean_f1' in locals() else None
            }

            # Add client results if provided
            if client_results:
                round_results["client_results"] = client_results

            # Add to results history
            self.results["rounds"].append(round_results)

        # Store history
        self.training_history["rounds"].append(self.round)
        self.training_history["global_test_loss"].append(test_loss)
        if accuracy is not None:
            self.training_history["global_test_accuracy"].append(accuracy)

        print(f"Round {self.round} - Global model test loss: {test_loss:.6f}")

        # Plot and save results
        self._save_results(predictions, actual)

        # Save experiment results
        self._save_experiment_results()

        return test_loss, accuracy

    def _read_client_results_from_files(self, round_num):
        """Read client training results from filesystem.

        Args:
            round_num: Round number to read client results from

        Returns:
            dict: Dictionary mapping client IDs to training results
        """
        if not self.storage_dir or not self.client_ids:
            return {}

        client_results = {}

        for client_id in self.client_ids:
            # Get client directory path
            round_dir = f"round_{round_num}"
            clients_dir = "clients"
            client_prefix = "client_"

            # Read results file
            client_output_dir = os.path.join(
                self.storage_dir,
                "output",
                "clients",
                f"{client_prefix}{client_id}"
            )
            results_path = os.path.join(client_output_dir, "training_results.json")

            # Read results if file exists
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    client_results[client_id] = results
                except Exception as e:
                    print(f"Error reading client {client_id} results: {e}")

        return client_results

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
                metric_plot_path = os.path.join(self.output_dir, "plots", f"rul_metrics_round_{self.round}.png")
            else:
                cm_plot_path = os.path.join(self.output_dir, "plots", f"mnist_confusion_matrix_round_{self.round}.png")
                acc_plot_path = os.path.join(self.output_dir, "plots", f"global_model_accuracy_round_{self.round}.png")
        else:
            history_path = f"output/server_results/training_history_round_{self.round}_{timestamp}.json"
            loss_plot_path = f"output/plots/global_model_loss_round_{self.round}_{timestamp}.png"

            if self.experiment_type == "n_cmapss":
                pred_plot_path = f"output/plots/rul_prediction_round_{self.round}_{timestamp}.png"
                metric_plot_path = f"output/plots/rul_metrics_round_{self.round}_{timestamp}.png"
            else:
                cm_plot_path = f"output/plots/mnist_confusion_matrix_round_{self.round}_{timestamp}.png"
                acc_plot_path = f"output/plots/global_model_accuracy_round_{self.round}_{timestamp}.png"

        # Convert numpy values to Python native types for JSON serialization
        history_for_json = self._make_json_serializable(self.training_history)

        # Save training history
        with open(history_path, "w") as f:
            json.dump(history_for_json, f)

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

            # Calculate error thresholds for coloring
            errors = predictions - actual
            within_10 = np.abs(errors) <= 10
            within_20 = np.logical_and(np.abs(errors) > 10, np.abs(errors) <= 20)
            beyond_20 = np.abs(errors) > 20

            # Create prediction scatter plot with colored points based on error
            plt.figure(figsize=(10, 6))

            # Plot points outside 20 cycles first (red)
            plt.scatter(actual[beyond_20], predictions[beyond_20], color='red', alpha=0.5, label='Error > 20 cycles')

            # Plot points within 10-20 cycles (yellow)
            plt.scatter(actual[within_20], predictions[within_20], color='orange', alpha=0.5, label='Error 10-20 cycles')

            # Plot points within 10 cycles last (green)
            plt.scatter(actual[within_10], predictions[within_10], color='green', alpha=0.5, label='Error ≤ 10 cycles')

            # Add perfect prediction line
            min_val = min(np.min(actual), np.min(predictions))
            max_val = max(np.max(actual), np.max(predictions))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

            # Add ±10 cycle lines
            plt.plot([min_val, max_val], [min_val + 10, max_val + 10], 'g--', alpha=0.3)
            plt.plot([min_val, max_val], [min_val - 10, max_val - 10], 'g--', alpha=0.3)

            # Add ±20 cycle lines
            plt.plot([min_val, max_val], [min_val + 20, max_val + 20], 'orange', linestyle='--', alpha=0.3)
            plt.plot([min_val, max_val], [min_val - 20, max_val - 20], 'orange', linestyle='--', alpha=0.3)

            plt.xlabel('Actual RUL (cycles)')
            plt.ylabel('Predicted RUL (cycles)')

            # Get current metrics
            rmse = self.training_history["global_test_loss"][-1]
            mae = self.training_history["rul_mae"][-1]
            r2 = self.training_history["rul_r_squared"][-1]
            within_10_pct = self.training_history["rul_within_10"][-1]
            within_20_pct = self.training_history["rul_within_20"][-1]

            plt.title(f'RUL Prediction - Round {self.round}\n'
                     f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}\n'
                     f'Within ±10 cycles: {within_10_pct:.2f}%, Within ±20 cycles: {within_20_pct:.2f}%')

            plt.legend()
            plt.grid(True)
            plt.savefig(pred_plot_path)
            plt.close()

            # Plot additional metrics across rounds if we have at least 2 rounds
            if len(self.training_history["rounds"]) >= 2:
                plt.figure(figsize=(15, 10))

                # Create a 2x2 grid of subplots
                plt.subplot(2, 2, 1)
                plt.plot(self.training_history["rounds"], self.training_history["global_test_loss"], marker='o')
                plt.xlabel('Federated Learning Round')
                plt.ylabel('RMSE (cycles)')
                plt.title('Root Mean Squared Error')
                plt.grid(True)

                plt.subplot(2, 2, 2)
                plt.plot(self.training_history["rounds"], self.training_history["rul_mae"], marker='o', color='orange')
                plt.xlabel('Federated Learning Round')
                plt.ylabel('MAE (cycles)')
                plt.title('Mean Absolute Error')
                plt.grid(True)

                plt.subplot(2, 2, 3)
                plt.plot(self.training_history["rounds"], self.training_history["rul_within_10"], marker='o', color='green')
                plt.xlabel('Federated Learning Round')
                plt.ylabel('Percentage (%)')
                plt.title('Predictions Within ±10 Cycles')
                plt.grid(True)

                plt.subplot(2, 2, 4)
                plt.plot(self.training_history["rounds"], self.training_history["rul_r_squared"], marker='o', color='purple')
                plt.xlabel('Federated Learning Round')
                plt.ylabel('R²')
                plt.title('Coefficient of Determination (R²)')
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(metric_plot_path)
                plt.close()

        # For MNIST, plot confusion matrix and accuracy
        elif self.experiment_type == "mnist":
            predictions = np.array(predictions)
            actual = np.array(actual)

            # Get the current metrics from the training history
            current_accuracy = self.training_history["global_test_accuracy"][-1]
            current_precision = self.training_history["mnist_precision"][-1]
            current_recall = self.training_history["mnist_recall"][-1]
            current_f1 = self.training_history["mnist_f1"][-1]
            current_per_class_acc = self.training_history["mnist_per_class_accuracy"][-1]

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(actual, predictions)
            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Create a figure with two subplots for raw and normalized confusion matrices
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Raw counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_xlabel('Predicted Labels')
            ax1.set_ylabel('True Labels')
            ax1.set_title('Confusion Matrix (Raw Counts)')

            # Normalized by row (true label)
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
            ax2.set_xlabel('Predicted Labels')
            ax2.set_ylabel('True Labels')
            ax2.set_title('Confusion Matrix (Normalized by True Label)')

            plt.suptitle(f'MNIST Classification Results - Round {self.round}')
            plt.tight_layout()
            plt.savefig(cm_plot_path)
            plt.close()

            # Plot metrics history if we have at least 2 rounds
            if len(self.training_history["rounds"]) >= 2:
                # Create a 2x2 subplot for accuracy, precision, recall, and F1 score
                plt.figure(figsize=(15, 10))

                # Accuracy
                plt.subplot(2, 2, 1)
                plt.plot(self.training_history["rounds"], self.training_history["global_test_accuracy"],
                         marker='o', color='blue')
                plt.xlabel('Federated Learning Round')
                plt.ylabel('Accuracy')
                plt.title('Overall Accuracy')
                plt.grid(True)

                # Precision
                plt.subplot(2, 2, 2)
                plt.plot(self.training_history["rounds"], self.training_history["mnist_precision"],
                         marker='o', color='green')
                plt.xlabel('Federated Learning Round')
                plt.ylabel('Precision')
                plt.title('Weighted Precision')
                plt.grid(True)

                # Recall
                plt.subplot(2, 2, 3)
                plt.plot(self.training_history["rounds"], self.training_history["mnist_recall"],
                         marker='o', color='orange')
                plt.xlabel('Federated Learning Round')
                plt.ylabel('Recall')
                plt.title('Weighted Recall')
                plt.grid(True)

                # F1 Score
                plt.subplot(2, 2, 4)
                plt.plot(self.training_history["rounds"], self.training_history["mnist_f1"],
                         marker='o', color='purple')
                plt.xlabel('Federated Learning Round')
                plt.ylabel('F1 Score')
                plt.title('Weighted F1 Score')
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(acc_plot_path)
                plt.close()

                # Plot per-class metrics for the current round
                num_classes = len(current_per_class_acc)
                plt.figure(figsize=(12, 6))

                # Retrieve per-class metrics for the latest round
                # We re-calculate precision, recall, and F1 scores per class
                precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                    actual, predictions, average=None, zero_division=0
                )

                classes = np.arange(num_classes)
                x = np.arange(len(classes))
                width = 0.2

                # Bar chart with per-class metrics
                plt.bar(x - 1.5*width, current_per_class_acc, width, label='Accuracy', color='blue')
                plt.bar(x - 0.5*width, precision_per_class, width, label='Precision', color='green')
                plt.bar(x + 0.5*width, recall_per_class, width, label='Recall', color='orange')
                plt.bar(x + 1.5*width, f1_per_class, width, label='F1 Score', color='purple')

                plt.xlabel('Class')
                plt.ylabel('Score')
                plt.title(f'Per-class Metrics - Round {self.round}')
                plt.xticks(x, classes)
                plt.legend()
                plt.grid(True, axis='y')

                # Add the current round's metrics as a subtitle
                plt.figtext(0.5, 0.01,
                           f"Overall: Acc={current_accuracy:.4f}, Prec={current_precision:.4f}, Rec={current_recall:.4f}, F1={current_f1:.4f}",
                           ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

                if self.storage_dir:
                    per_class_path = os.path.join(self.output_dir, "plots", f"mnist_per_class_metrics_round_{self.round}.png")
                else:
                    per_class_path = f"output/plots/mnist_per_class_metrics_round_{self.round}_{timestamp}.png"

                plt.tight_layout()
                plt.savefig(per_class_path)
                plt.close()

        # Save model
        if self.storage_dir:
            models_dir = os.path.join(self.storage_dir, "output", "global_models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"global_model_round_{self.round}.pt")
        else:
            os.makedirs("output/global_models", exist_ok=True)
            model_path = f'output/global_models/{self.experiment_type}_global_model_round_{self.round}_{timestamp}.pt'

        torch.save(self.global_model.state_dict(), model_path)

        print(f"Saved results for round {self.round}")

    def _save_experiment_results(self):
        """Save experiment results to a JSON file."""
        # Check if we have a storage directory
        if not self.storage_dir:
            return

        # Create the output directory if it doesn't exist
        output_dir = os.path.join(self.storage_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Convert NumPy types to Python native types
        serializable_results = self._make_json_serializable(self.results)

        # Save the results
        results_path = os.path.join(output_dir, "fl_results.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Saved experiment results to {results_path}")

    def _make_json_serializable(self, obj):
        """Convert an object with potential NumPy types to JSON serializable format.

        Args:
            obj: The object to convert

        Returns:
            Object with all NumPy types converted to Python native types
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        elif hasattr(obj, 'dtype') and np.isscalar(obj):  # Catch any other NumPy scalar types
            return obj.item()
        else:
            return obj

    def initialize_model(self, round_num=0):
        """Initialize and save the initial model.

        Args:
            round_num: Usually 0 for the initial model
        """
        if not self.storage_dir:
            return

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Create the initial model directory
        initial_model_dir = os.path.join(self.storage_dir, structure["global_model_initial"])
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
        if not self.storage_dir:
            return

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Create round directory
        round_dir = os.path.join(
            self.storage_dir,
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
            source_model_dir = os.path.join(self.storage_dir, structure["global_model_initial"])
        else:
            # Use the previous round's aggregated model
            prev_round_dir = os.path.join(
                self.storage_dir,
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
        if not self.storage_dir:
            return False

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Get the round directory
        round_dir = os.path.join(
            self.storage_dir,
            structure["round_template"].format(round=round_num)
        )

        # Get the clients directory
        clients_dir = os.path.join(round_dir, structure["clients_dir"])

        # Get the aggregated model directory
        aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
        os.makedirs(aggregated_model_dir, exist_ok=True)

        # Aggregate client models
        self.aggregate_models_from_files(clients_dir)

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
