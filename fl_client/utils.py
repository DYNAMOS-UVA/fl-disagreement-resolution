"""Utility functions for federated learning client."""

import os
import json
from datetime import datetime

def save_training_results(client, results):
    """Save training results to a JSON file.

    Args:
        client: FederatedClient instance
        results: Dictionary containing training results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results to output directory
    if client.storage_dir:
        # Make sure the output directory exists
        output_dir = os.path.join(client.storage_dir, "output", "clients", f"client_{client.client_id}")
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "training_results.json")
    else:
        os.makedirs("output/client_results", exist_ok=True)
        results_path = f"output/client_results/client_{client.client_id}_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Client {client.client_id} saved training results to {results_path}")

def get_structure_config(storage_dir):
    """Get the directory structure configuration.

    Args:
        storage_dir: Storage directory for models and results

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
    config_path = os.path.join(os.path.dirname(storage_dir), "mock_etcd/configuration.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "storage" in config and "structure" in config["storage"]:
                    return config["storage"]["structure"]
    except Exception as e:
        print(f"Error loading configuration: {e}")

    return default_structure
