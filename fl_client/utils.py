"""Utility functions for federated learning client."""

import os
import json
from datetime import datetime

def save_training_results(client, results):
    """Save training results to the output directory.

    Args:
        client: The client object
        results: The training results to save
    """
    # If we have a proper output directory
    if client.results_dir:
        # Create the client output directory
        output_dir = os.path.join(client.results_dir, "output", "clients", f"client_{client.client_id}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Fallback to legacy directory
        output_dir = "output/client_results"
        os.makedirs(output_dir, exist_ok=True)

    # Save results as JSON
    results_path = os.path.join(output_dir, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Client {client.client_id} saved training results to {results_path}")

def get_structure_config(results_dir):
    """Get the directory structure configuration.

    Args:
        results_dir: Results directory for models and results

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
    config_path = os.path.join(os.path.dirname(results_dir), "mock_etcd/configuration.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "results" in config and "structure" in config["results"]:
                    return config["results"]["structure"]
    except Exception as e:
        print(f"Error loading configuration: {e}")

    return default_structure
