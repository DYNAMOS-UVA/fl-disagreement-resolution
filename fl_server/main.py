"""Command-line interface for federated learning server."""

import argparse
from fl_server.server import FederatedServer

def main():
    """Run the server as a standalone application."""
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
