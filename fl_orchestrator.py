"""Federated learning orchestrator implementation."""

import os
import argparse
import json
import fl_module

from fl_client import FederatedClient
from fl_server import FederatedServer
from mock_etcd.etcd_loader import MockEtcdLoader

class FederatedOrchestrator:
    """Orchestrator for coordinating clients and server in federated learning."""

    def __init__(self, config_path="mock_etcd/configuration.json"):
        """Initialize the federated learning orchestrator.

        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = MockEtcdLoader(config_path)

        # Get configuration sections
        self.exp_config = self.config.get_experiment_config()
        self.data_config = self.config.get_data_config()
        self.train_config = self.config.get_training_config()
        self.results_config = self.config.get_results_config()

        # Extract common parameters
        self.experiment_type = self.exp_config.get("type")
        self.client_ids_in_experiment = self.exp_config.get("client_ids")
        self.fl_rounds = self.exp_config.get("fl_rounds")
        self.results_dir = self.results_config.get("results_dir")

        # Setup MNIST data if needed
        self._setup_data_if_needed()

        # Create model_storage directory
        if self.results_dir:
            structure = self.results_config.get("structure", {})
            model_storage_dir = structure.get("model_storage_dir", "model_storage")
            os.makedirs(os.path.join(self.results_dir, model_storage_dir), exist_ok=True)

        # Initialize server
        self.server = self._init_server()

        # Initialize clients
        self.clients = self._init_clients()

        print(f"Initialized federated learning orchestrator with {len(self.client_ids_in_experiment)} clients for {self.experiment_type} experiment")
        print(f"Results directory: {self.results_dir}")

    def _setup_data_if_needed(self):
        """Setup data if needed based on configuration."""
        # Check if MNIST data needs to be setup
        if self.experiment_type == "mnist" and self.data_config.get("setup_data", False):
            # Check if data already exists for all clients
            train_client_data_exists = all(
                os.path.exists(os.path.join("data/mnist", 'train', f'client_{i}', 'mnist_data.npz'))
                for i in range(max(self.client_ids_in_experiment) + 1)
            )
            test_data_exists = os.path.exists(os.path.join("data/mnist", 'test', 'mnist_test.npz'))

            # Setup data if needed
            if not (train_client_data_exists and test_data_exists) or self.data_config.get("force_setup_data", False):
                print("Setting up MNIST federated data...")
                fl_module.setup_mnist_federated_data(
                    num_clients=max(self.client_ids_in_experiment) + 1,  # Ensure enough clients are created
                    samples_per_client=self.data_config.get("client_sample_size", 1000),
                    iid=self.exp_config.get("iid", True)
                )
                print("MNIST data setup complete.")
            else:
                print("MNIST data already exists. Skipping setup.")
                print("Use --force_setup_data to force recreation.")

    def _init_server(self):
        """Initialize the federated learning server.

        Returns:
            FederatedServer: Initialized server
        """
        server = FederatedServer(
            experiment_type=self.experiment_type,
            test_dir=self.config.get_test_dir(),
            test_units=self.data_config.get("test_units"),
            results_dir=self.results_dir
        )

        # Load test data for evaluation
        if self.experiment_type == "n_cmapss":
            server.load_test_data(sample_size=self.data_config.get("test_sample_size", 500))
        else:
            server.load_test_data()

        # Initialize the server with experiment metadata
        server.init_experiment(
            fl_rounds=self.fl_rounds,
            client_ids=self.client_ids_in_experiment,
            iid=self.exp_config.get("iid", False) if self.experiment_type == "mnist" else None
        )

        return server

    def _init_clients(self):
        """Initialize federated learning clients.

        Returns:
            dict: Dictionary mapping client IDs to client instances
        """
        clients = {}
        for client_id in self.client_ids_in_experiment:
            clients[client_id] = FederatedClient(
                client_id=client_id,
                experiment_type=self.experiment_type,
                data_dir=self.config.get_train_dir(),
                batch_size=self.train_config.get("batch_size", 64),
                epochs=self.train_config.get("local_epochs", 5),
                learning_rate=self.train_config.get("learning_rate", 0.001),
                results_dir=self.results_dir
            )
            # Load client data
            clients[client_id].load_data(sample_size=self.data_config.get("client_sample_size", 1000))

        return clients

    def run_federated_learning(self):
        """Execute federated learning process for specified number of rounds.

        The federated learning process follows these steps:
        1. Initialize the model (global_model_initial)
        2. For each round:
           a. Copy the latest aggregated model (or initial model for round 1) to global_model_for_training
           b. Distribute global_model_for_training to clients for local training
           c. Clients train on local data and save their models to their directories
           d. Server aggregates client models into global_model_aggregated
           e. Server evaluates the aggregated model
        """
        print(f"Starting federated learning with {self.fl_rounds} rounds...")

        # Initialize and save the initial global model
        self.server.initialize_model(round_num=0)

        # Initial evaluation of the global model (round 0)
        self.server.evaluate_model(fl_round=0)
        print("Initial model evaluation completed")

        # Main federated learning loop
        for fl_round in range(1, self.fl_rounds + 1):
            print(f"\n--- Federated Learning Round {fl_round}/{self.fl_rounds} ---")

            # 1. Server prepares the global model for this round
            print("Preparing global model for training...")

            # Server creates directories and prepares model for this round
            if fl_round == 1:
                # For the first round, use the initial model
                self.server.prepare_training_model(fl_round, use_initial=True)
                print("Using global_model_initial for round 1")
            else:
                # For subsequent rounds, use the previous round's aggregated model
                self.server.prepare_training_model(fl_round, use_initial=False)
                print(f"Using global_model_aggregated from round {fl_round-1}")

            # 2. Train local models on each client
            print("Training local models...")

            # Get fully excluded clients from the server
            fully_excluded_clients_for_round = self.server.fully_excluded_clients_for_current_round

            if fully_excluded_clients_for_round:
                print(f"Orchestrator: Fully excluded clients for round {fl_round}: {sorted(list(fully_excluded_clients_for_round))}")

            for client_id in self.client_ids_in_experiment:
                if client_id in fully_excluded_clients_for_round:
                    print(f"Skipping training for client {client_id} in round {fl_round} due to full exclusion.")
                    continue

                if client_id not in self.clients:
                    print(f"Warning: Client {client_id} configured in experiment but not initialized. Skipping.")
                    continue

                client = self.clients[client_id]
                print(f"Training client {client_id}...")

                # Client loads the global model for this round
                client.load_round_model(fl_round)

                # Client trains model and saves results to filesystem
                client.train(epochs=self.train_config.get("local_epochs", 5), round_num=fl_round)

                # Client saves trained model to filesystem
                client.save_round_model(fl_round)

            # 3. Server aggregates models from all clients
            print("Aggregating client models...")

            # Server loads client models and aggregates them
            self.server.aggregate_client_models(fl_round)

            # 4. Server evaluates the aggregated model
            print("Evaluating aggregated model...")

            # Server evaluates the model and stores metrics
            self.server.evaluate_model(fl_round=fl_round)

            print(f"Round {fl_round} completed.")

        print("\nFederated learning completed!")


def main():
    """Run the orchestrator as a standalone application."""
    parser = argparse.ArgumentParser(description="Federated Learning Orchestrator")
    parser.add_argument("--config", type=str, default="mock_etcd/configuration.json",
                        help="Path to configuration file")
    parser.add_argument("--override", action="store_true",
                        help="Override configuration with command line arguments")

    # Optional override arguments (only used if --override is specified)
    parser.add_argument("--experiment", type=str, choices=["n_cmapss", "mnist"],
                        help="Experiment type")
    parser.add_argument("--clients", type=int, nargs="+",
                        help="Client IDs")
    parser.add_argument("--fl_rounds", type=int,
                        help="Number of federated learning rounds")
    parser.add_argument("--local_epochs", type=int,
                        help="Number of local training epochs")
    parser.add_argument("--setup_data", action="store_true",
                        help="Set up experiment data (for MNIST)")
    parser.add_argument("--force_setup_data", action="store_true",
                        help="Force data setup even if it exists")
    parser.add_argument("--iid", action="store_true",
                        help="Use IID data distribution (for MNIST)")
    parser.add_argument("--results_dir", type=str,
                        help="Results directory for models and outputs")

    args = parser.parse_args()

    # If override is specified, update the configuration file with command line arguments
    if args.override:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)

            # Update configuration with command line arguments
            if args.experiment:
                config["experiment"]["type"] = args.experiment
            if args.clients:
                config["experiment"]["client_ids"] = args.clients
            if args.fl_rounds:
                config["experiment"]["fl_rounds"] = args.fl_rounds
            if args.local_epochs:
                config["training"]["local_epochs"] = args.local_epochs
            if args.setup_data:
                config["data"]["setup_data"] = True
            if args.force_setup_data:
                config["data"]["force_setup_data"] = True
            if args.iid:
                config["experiment"]["iid"] = True
            if args.results_dir:
                config["results"]["custom_dir"] = args.results_dir

            # Write updated configuration back to file
            with open(args.config, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"Updated configuration in {args.config}")
        except Exception as e:
            print(f"Error updating configuration: {e}")
            print("Using original configuration")

    # Create and run orchestrator
    orchestrator = FederatedOrchestrator(config_path=args.config)

    # Run federated learning
    orchestrator.run_federated_learning()


if __name__ == "__main__":
    main()
