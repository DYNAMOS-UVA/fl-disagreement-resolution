"""Federated learning orchestrator implementation."""

import os
import argparse
import json
import time
import shutil
from datetime import datetime

from fl_client import FederatedClient
from fl_server import FederatedServer
import data_module

class FederatedOrchestrator:
    """Orchestrator for coordinating clients and server in federated learning."""

    def __init__(
        self,
        experiment_type,
        clients,
        train_dir=None,
        test_dir=None,
        test_units=None,
        client_sample_size=1000,
        test_sample_size=500,
        batch_size=64,
        local_epochs=5,
        learning_rate=0.001,
        fl_rounds=3,
        setup_data=False,
        force_setup_data=False,
        iid=True,
        storage_dir=None
    ):
        """Initialize the federated learning orchestrator.

        Args:
            experiment_type: Type of experiment ('n_cmapss' or 'mnist')
            clients: List of client IDs to include in the experiment
            train_dir: Directory containing training data
            test_dir: Directory containing test data
            test_units: List of unit IDs to use for testing (for N-CMAPSS)
            client_sample_size: Maximum number of samples to load per client
            test_sample_size: Maximum number of samples to load per test unit
            batch_size: Batch size for training and testing
            local_epochs: Number of local training epochs per round
            learning_rate: Learning rate for optimization
            fl_rounds: Number of federated learning rounds
            setup_data: Whether to set up experiment data
            force_setup_data: Whether to force data setup even if it exists
            iid: Whether to use IID data distribution (for MNIST)
            storage_dir: Custom storage directory path
        """
        self.experiment_type = experiment_type
        self.client_ids = clients
        self.client_sample_size = client_sample_size
        self.test_sample_size = test_sample_size
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.fl_rounds = fl_rounds
        self.setup_data = setup_data
        self.force_setup_data = force_setup_data
        self.iid = iid

        # Create simulation ID based on timestamp
        self.simulation_id = f"fl_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create storage directories
        if storage_dir:
            self.storage_dir = storage_dir
        else:
            self.storage_dir = os.path.join("storage", self.simulation_id)

        os.makedirs(self.storage_dir, exist_ok=True)

        # Create output directory for this simulation
        self.output_dir = os.path.join(self.storage_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Set default directories based on experiment type
        if train_dir is None:
            if experiment_type == "n_cmapss":
                self.train_dir = "data/n-cmapss/train"
            elif experiment_type == "mnist":
                self.train_dir = "data/mnist/train"
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")
        else:
            self.train_dir = train_dir

        if test_dir is None:
            if experiment_type == "n_cmapss":
                self.test_dir = "data/n-cmapss/test"
            elif experiment_type == "mnist":
                self.test_dir = "data/mnist/test"
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")
        else:
            self.test_dir = test_dir

        self.test_units = test_units

        # Check if MNIST data needs to be setup
        if self.experiment_type == "mnist":
            # Check if data already exists for all clients
            train_client_data_exists = all(
                os.path.exists(os.path.join("data/mnist", 'train', f'client_{i}', 'mnist_data.npz'))
                for i in range(max(self.client_ids) + 1)
            )
            test_data_exists = os.path.exists(os.path.join("data/mnist", 'test', 'mnist_test.npz'))

            # Setup data if needed
            if self.setup_data and (not (train_client_data_exists and test_data_exists) or self.force_setup_data):
                print("Setting up MNIST federated data...")
                data_module.setup_mnist_federated_data(
                    num_clients=max(self.client_ids) + 1,  # Ensure enough clients are created
                    samples_per_client=client_sample_size,
                    iid=self.iid
                )
                print("MNIST data setup complete.")
            elif self.setup_data:
                print("MNIST data already exists. Skipping setup.")
                print("Use --force_setup_data to force recreation.")

        # Initialize server
        self.server = FederatedServer(
            experiment_type=experiment_type,
            test_dir=self.test_dir,
            test_units=self.test_units,
            storage_dir=self.storage_dir
        )

        # Load test data for evaluation
        if self.experiment_type == "n_cmapss":
            self.server.load_test_data(sample_size=test_sample_size)
        else:
            self.server.load_test_data()

        # Initialize clients
        self.clients = {}
        for client_id in self.client_ids:
            self.clients[client_id] = FederatedClient(
                client_id=client_id,
                experiment_type=experiment_type,
                data_dir=self.train_dir,
                batch_size=batch_size,
                epochs=local_epochs,
                learning_rate=learning_rate,
                storage_dir=self.storage_dir
            )
            # Load client data
            self.clients[client_id].load_data(sample_size=client_sample_size)

        # Results tracking
        self.results = {
            "experiment_type": experiment_type,
            "fl_rounds": fl_rounds,
            "client_ids": clients,
            "iid": self.iid if self.experiment_type == "mnist" else None,
            "rounds": []
        }

        print(f"Initialized federated learning orchestrator with {len(clients)} clients for {experiment_type} experiment")
        print(f"Storage directory: {self.storage_dir}")

    def run_federated_learning(self):
        """Execute federated learning process for specified number of rounds.

        Returns:
            dict: Results of the federated learning process
        """
        print(f"Starting federated learning with {self.fl_rounds} rounds...")

        # Create initial global model directory
        initial_model_dir = os.path.join(self.storage_dir, "global_model_initial")
        os.makedirs(initial_model_dir, exist_ok=True)

        # Save initial global model
        self.server.save_model(initial_model_dir)

        # Initial evaluation of the global model
        initial_test_loss, initial_accuracy = self.server.evaluate_model()
        print(f"Initial global model test loss: {initial_test_loss:.6f}")
        if initial_accuracy is not None:
            print(f"Initial global model test accuracy: {initial_accuracy:.4f}")

        # Main federated learning loop
        for fl_round in range(1, self.fl_rounds + 1):
            print(f"\n--- Federated Learning Round {fl_round}/{self.fl_rounds} ---")

            # Create round directory
            round_dir = os.path.join(self.storage_dir, f"round_{fl_round}")
            os.makedirs(round_dir, exist_ok=True)

            # Create directories for global and client models
            global_model_dir = os.path.join(round_dir, "global_model")
            os.makedirs(global_model_dir, exist_ok=True)

            clients_dir = os.path.join(round_dir, "clients")
            os.makedirs(clients_dir, exist_ok=True)

            # 1. Distribute global model to all clients
            # First, copy the global model to the current round directory
            if fl_round == 1:
                # Use initial model for first round
                global_model_path = initial_model_dir
            else:
                # Use previous round's aggregated model
                prev_round_dir = os.path.join(self.storage_dir, f"round_{fl_round-1}")
                prev_global_model_dir = os.path.join(prev_round_dir, "global_model_aggregated")
                global_model_path = prev_global_model_dir

            # Save current global model to this round's directory
            self.server.load_model(global_model_path)
            self.server.save_model(global_model_dir)

            # 2. Train local models on each client
            print("Training local models...")
            client_train_losses = {}
            client_valid_losses = {}

            for client_id, client in self.clients.items():
                print(f"Training client {client_id}...")
                client_dir = os.path.join(clients_dir, f"client_{client_id}")
                os.makedirs(client_dir, exist_ok=True)

                # Load global model
                client.load_model(global_model_dir)

                # Train model
                train_losses, valid_losses = client.train(epochs=self.local_epochs)

                # Save trained model
                client.save_model(client_dir)

                client_train_losses[client_id] = train_losses
                client_valid_losses[client_id] = valid_losses

            # 3. Aggregate models
            print("Aggregating models...")
            aggregated_model_dir = os.path.join(round_dir, "global_model_aggregated")
            os.makedirs(aggregated_model_dir, exist_ok=True)

            # Server loads and aggregates client models
            self.server.aggregate_models_from_files(clients_dir)

            # Save aggregated model
            self.server.save_model(aggregated_model_dir)

            # 4. Evaluate global model
            print("Evaluating global model...")
            test_loss, accuracy = self.server.evaluate_model()

            # 5. Save round results
            round_results = {
                "round": fl_round,
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "client_train_losses": client_train_losses,
                "client_valid_losses": client_valid_losses
            }
            self.results["rounds"].append(round_results)

            # Save results after each round
            self._save_results()

            print(f"Round {fl_round} completed. Global model test loss: {test_loss:.6f}")
            if accuracy is not None:
                print(f"Global model test accuracy: {accuracy:.4f}")

        print("\nFederated learning completed!")
        return self.results

    def _save_results(self):
        """Save orchestrator results to a JSON file."""
        results_path = os.path.join(self.output_dir, "fl_results.json")

        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Saved orchestrator results to {results_path}")


def main():
    """Run the orchestrator as a standalone application."""
    parser = argparse.ArgumentParser(description="Federated Learning Orchestrator")
    parser.add_argument("--experiment", type=str, default="n_cmapss", choices=["n_cmapss", "mnist"], help="Experiment type")
    parser.add_argument("--clients", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5], help="Client IDs")
    parser.add_argument("--train_dir", type=str, help="Training data directory (defaults to experiment-specific location)")
    parser.add_argument("--test_dir", type=str, help="Test data directory (defaults to experiment-specific location)")
    parser.add_argument("--test_units", type=int, nargs="+", default=[11, 14, 15], help="Test units (for N-CMAPSS)")
    parser.add_argument("--client_sample_size", type=int, default=1000, help="Sample size per client")
    parser.add_argument("--test_sample_size", type=int, default=500, help="Sample size per test unit (for N-CMAPSS)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--local_epochs", type=int, default=5, help="Number of local training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--fl_rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--setup_data", action="store_true", help="Set up experiment data (for MNIST)")
    parser.add_argument("--force_setup_data", action="store_true", help="Force data setup even if it exists")
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution (for MNIST)")
    parser.add_argument("--storage_dir", type=str, help="Storage directory for models and results")

    args = parser.parse_args()

    # Create and run orchestrator
    orchestrator = FederatedOrchestrator(
        experiment_type=args.experiment,
        clients=args.clients,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        test_units=args.test_units if args.experiment == "n_cmapss" else None,
        client_sample_size=args.client_sample_size,
        test_sample_size=args.test_sample_size,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        fl_rounds=args.fl_rounds,
        setup_data=args.setup_data,
        force_setup_data=args.force_setup_data,
        iid=args.iid,
        storage_dir=args.storage_dir
    )

    # Run federated learning
    orchestrator.run_federated_learning()


if __name__ == "__main__":
    main()
