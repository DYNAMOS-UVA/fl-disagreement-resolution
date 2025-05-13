import os
import argparse
import json
from datetime import datetime

from fl_client import FederatedClient
from fl_server import FederatedServer
from data_utils import setup_mnist_federated_data

class FederatedOrchestrator:
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
        iid=True
    ):
        self.experiment_type = experiment_type
        self.client_ids = clients
        self.client_sample_size = client_sample_size
        self.test_sample_size = test_sample_size
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.fl_rounds = fl_rounds
        self.setup_data = setup_data
        self.iid = iid

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

        # Prepare output directories
        os.makedirs("output/orchestrator_results", exist_ok=True)

        # Setup MNIST data if needed
        if self.experiment_type == "mnist" and self.setup_data:
            print("Setting up MNIST federated data...")
            setup_mnist_federated_data(
                num_clients=len(self.client_ids),
                samples_per_client=client_sample_size,
                iid=self.iid
            )
            print("MNIST data setup complete.")

        # Initialize server
        self.server = FederatedServer(
            experiment_type=experiment_type,
            test_dir=self.test_dir,
            test_units=self.test_units
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
                learning_rate=learning_rate
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

    def run_federated_learning(self):
        """Execute federated learning process for specified number of rounds"""
        print(f"Starting federated learning with {self.fl_rounds} rounds...")

        # Initial evaluation of the global model
        initial_test_loss, initial_accuracy = self.server.evaluate_model()
        print(f"Initial global model test loss: {initial_test_loss:.6f}")
        if initial_accuracy is not None:
            print(f"Initial global model test accuracy: {initial_accuracy:.4f}")

        # Main federated learning loop
        for fl_round in range(1, self.fl_rounds + 1):
            print(f"\n--- Federated Learning Round {fl_round}/{self.fl_rounds} ---")

            # 1. Distribute global model to all clients
            global_parameters = self.server.get_model_parameters()
            for client_id, client in self.clients.items():
                client.set_model_parameters(global_parameters)

            # 2. Train local models on each client
            print("Training local models...")
            client_train_losses = {}
            client_valid_losses = {}

            for client_id, client in self.clients.items():
                print(f"Training client {client_id}...")
                train_losses, valid_losses = client.train(epochs=self.local_epochs)
                client_train_losses[client_id] = train_losses
                client_valid_losses[client_id] = valid_losses

            # 3. Collect updated models from clients
            print("Collecting client models...")
            client_parameters = {}
            for client_id, client in self.clients.items():
                client_parameters[client_id] = client.get_model_parameters()

            # 4. Aggregate models (using equal weighting)
            print("Aggregating models...")
            self.server.aggregate_models(client_parameters)

            # 5. Evaluate global model
            print("Evaluating global model...")
            test_loss, accuracy = self.server.evaluate_model()

            # 6. Save round results
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
        """Save orchestrator results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"output/orchestrator_results/fl_results_{self.experiment_type}_{timestamp}.json"

        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Saved orchestrator results to {results_path}")

def main():
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
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution (for MNIST)")

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
        iid=args.iid
    )

    # Run federated learning
    orchestrator.run_federated_learning()

if __name__ == "__main__":
    main()
