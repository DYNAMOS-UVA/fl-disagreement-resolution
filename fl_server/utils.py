"""Utility functions for federated learning server."""

import os
import json
import numpy as np

def make_json_serializable(obj):
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
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif hasattr(obj, 'dtype') and np.isscalar(obj):  # Catch any other NumPy scalar types
        return obj.item()
    else:
        return obj

def read_client_results_from_files(storage_dir, client_ids, round_num):
    """Read client training results from filesystem.

    Args:
        storage_dir: Base storage directory
        client_ids: List of client IDs to read results for
        round_num: Round number to read client results from

    Returns:
        dict: Dictionary mapping client IDs to training results
    """
    if not storage_dir or not client_ids:
        return {}

    client_results = {}

    for client_id in client_ids:
        # Get client directory path
        client_prefix = "client_"

        # Read results file
        client_output_dir = os.path.join(
            storage_dir,
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
