"""Federated learning orchestrator implementation."""

import os
import argparse
import json
import time
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

        print(f"Initialized disagreement-aware federated learning orchestrator")
        print(f"  - {len(self.client_ids_in_experiment)} clients for {self.experiment_type} experiment")
        print(f"  - Advanced disagreement resolution with multi-track model training")
        print(f"  - Results directory: {self.results_dir}")

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
            results_dir=self.results_dir,
            verbose_plots=self.results_config.get("verbose_plots", False)
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
        """Execute federated learning with sophisticated disagreement resolution.

        This process implements a disagreement-aware federated learning system that:
        1. Analyzes active client disagreements for each round
        2. Creates separate model tracks for conflicting client groups
        3. Enables clients to train on multiple tracks (primary + background participation)
        4. Performs track-based aggregation with optional deep rewind/incremental finetuning
        5. Evaluates both global and track-specific model performance

        The process follows these steps:
        1. Initialize the global model (global_model_initial)
        2. For each round:
           a. Server analyzes disagreements and prepares track-specific models
           b. Clients load their assigned primary track model + any background track models
           c. Clients train on multiple tracks based on disagreement participation
           d. Server aggregates models using disagreement-aware track algorithm
           e. Server evaluates global model and individual track performance
        """
        # Start timing the entire federated learning process
        fl_start_time = time.time()

        print(f"Starting federated learning with {self.fl_rounds} rounds...")

        # Initialize and save the initial global model
        self.server.initialize_model(round_num=0)

        # Initial evaluation of the global model (round 0)
        self.server.evaluate_model(fl_round=0)
        print("Initial model evaluation completed")

        # Initialize round timing history
        if not hasattr(self.server, 'round_timing_history'):
            self.server.round_timing_history = []

        # Main federated learning loop
        for fl_round in range(1, self.fl_rounds + 1):
            # Start timing the entire round
            round_start_time = time.time()

            print(f"\n--- Federated Learning Round {fl_round}/{self.fl_rounds} ---")

            # 1. Server analyzes disagreements and prepares track-specific models
            print("Analyzing disagreements and preparing track-specific models...")

            # Server creates directories and prepares models with disagreement resolution
            if fl_round == 1:
                # For the first round, create initial tracks from global model
                training_model_dir, track_init_time = self.server.prepare_training_model(fl_round, use_initial=True)
                print("Created initial track models from global_model_initial for round 1")
            else:
                # For subsequent rounds, update tracks based on disagreement evolution
                training_model_dir, track_init_time = self.server.prepare_training_model(fl_round, use_initial=False)
                print(f"Updated track models based on disagreement changes from round {fl_round-1}")

            # 2. Clients participate in disagreement-aware multi-track training
            print("Starting disagreement-aware multi-track client training...")

            # Start timing client training phase
            client_training_start_time = time.time()
            client_training_times = {}

            # Get fully excluded clients from the server
            fully_excluded_clients_for_round = self.server.fully_excluded_clients_for_current_round

            if fully_excluded_clients_for_round:
                print(f"Orchestrator: Fully excluded clients for round {fl_round}: {sorted(list(fully_excluded_clients_for_round))}")

            for client_id in self.client_ids_in_experiment:
                if client_id in fully_excluded_clients_for_round:
                    print(f"Skipping training for client {client_id} in round {fl_round} due to full exclusion from all tracks.")
                    continue

                if client_id not in self.clients:
                    print(f"Warning: Client {client_id} configured in experiment but not initialized. Skipping.")
                    continue

                client = self.clients[client_id]
                print(f"Client {client_id}: Loading track models and training with disagreement resolution...")

                # Time individual client training
                client_start_time = time.time()

                # Client loads primary track model and any background track models
                client.load_track_models_for_round(fl_round)

                # Client trains on primary track + participates in background tracks
                training_results = client.train_with_disagreement_resolution(epochs=self.train_config.get("local_epochs", 5), round_num=fl_round)

                # Client saves all trained models (primary + background) to filesystem
                client.save_trained_track_models(fl_round)

                # Record individual client training time
                client_training_time = time.time() - client_start_time
                client_training_times[client_id] = {
                    "training_time_seconds": client_training_time,
                    "epochs": self.train_config.get("local_epochs", 5),
                    "total_training_time_from_results": training_results.get("training_time", {}).get("total_seconds", 0) if training_results else 0
                }
                print(f"Client {client_id} completed training in {client_training_time:.4f} seconds")

            # Calculate total client training phase time
            total_client_training_time = time.time() - client_training_start_time

            # 3. Server performs disagreement-aware track-based aggregation
            print("Performing disagreement-aware track-based model aggregation...")

            # Server aggregates models using sophisticated track algorithm
            self.server.aggregate_with_disagreement_resolution(fl_round)

            # 4. Server evaluates global model and individual track performance
            print("Evaluating global model and track-specific performance...")

            # Server evaluates both global and track models, storing comprehensive metrics
            self.server.evaluate_model(fl_round=fl_round)

            # Calculate total round time
            total_round_time = time.time() - round_start_time

            # Extract aggregation timing data for this round
            aggregation_timing = None
            if hasattr(self.server, 'aggregation_timing_history') and self.server.aggregation_timing_history:
                # Get the most recent aggregation timing entry (should be for this round)
                for timing_entry in reversed(self.server.aggregation_timing_history):
                    if timing_entry.get("round") == fl_round:
                        aggregation_timing = timing_entry
                        break

            # Store round timing metrics
            round_timing = {
                "round": fl_round,
                "track_model_initialization_time_seconds": track_init_time,
                "client_training_times": client_training_times,
                "total_client_training_time_seconds": total_client_training_time,
                "total_round_time_seconds": total_round_time,
                "num_participating_clients": len(client_training_times)
            }

            # Add aggregation timing data if available
            if aggregation_timing:
                round_timing.update({
                    "aggregation_time_seconds": aggregation_timing.get("aggregation_time_seconds", 0),
                    "resolution_time_seconds": aggregation_timing.get("resolution_time_seconds", 0),
                    "total_aggregation_time_seconds": aggregation_timing.get("total_aggregation_time_seconds", 0),
                    "has_disagreements": aggregation_timing.get("has_disagreements", False)
                })

            self.server.round_timing_history.append(round_timing)

            print(f"Round {fl_round} completed with disagreement resolution.")
            print(f"Round {fl_round} timing summary:")
            print(f"  Track initialization: {track_init_time:.4f}s")
            print(f"  Client training phase: {total_client_training_time:.4f}s")
            if aggregation_timing:
                print(f"  Resolution time: {aggregation_timing.get('resolution_time_seconds', 0):.4f}s")
                print(f"  Aggregation time: {aggregation_timing.get('aggregation_time_seconds', 0):.4f}s")
                print(f"  Total aggregation time: {aggregation_timing.get('total_aggregation_time_seconds', 0):.4f}s")
            print(f"  Total round time: {total_round_time:.4f}s")

        # Calculate total running time
        total_running_time = time.time() - fl_start_time

        # Add timing information to server results and save final results
        self.server.set_total_running_time(total_running_time)
        self.server._save_experiment_results()

        print(f"\nFederated learning with disagreement resolution completed!")
        print(f"Total running time: {total_running_time:.2f} seconds")


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
    parser.add_argument("--verbose_plots", action="store_true",
                        help="Generate all plots (default: only last round track metrics + track contributions)")

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
            if args.verbose_plots:
                config["results"]["verbose_plots"] = True

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
