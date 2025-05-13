import argparse
from data_utils import setup_mnist_federated_data

def main():
    parser = argparse.ArgumentParser(description="Setup MNIST data for federated learning")
    parser.add_argument("--num_clients", type=int, default=6, help="Number of clients")
    parser.add_argument("--samples_per_client", type=int, default=1000, help="Number of samples per client")
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution (default is Non-IID)")
    parser.add_argument("--data_dir", type=str, default="data/mnist", help="Data directory")

    args = parser.parse_args()

    print(f"Setting up MNIST data for {args.num_clients} clients with {'IID' if args.iid else 'Non-IID'} distribution")
    print(f"Each client will have approximately {args.samples_per_client} samples")

    setup_mnist_federated_data(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        iid=args.iid,
        data_dir=args.data_dir
    )

    print("MNIST data setup complete!")
    print(f"Data saved to {args.data_dir}")
    print("\nYou can now run federated learning with:")
    print(f"python fl_orchestrator.py --experiment mnist --clients {' '.join(str(i) for i in range(args.num_clients))} --fl_rounds 3 {'--iid' if args.iid else ''}")

if __name__ == "__main__":
    main()
