"""Utility functions for federated learning server."""

import os
import json
import numpy as np
from datetime import datetime
from fl_client.utils import get_structure_config

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

def read_client_results_from_files(results_dir, client_ids, round_num):
    """Read client training results from filesystem for a specific round.

    Args:
        results_dir: Base results directory
        client_ids: List of client IDs
        round_num: Round number

    Returns:
        dict: Dictionary mapping client IDs to their training results
    """
    if not results_dir or not client_ids:
        return {}

    client_results = {}

    # Construct round directory path
    structure = get_structure_config(results_dir)
    round_dir = os.path.join(
        results_dir,
        structure["round_template"].format(round=round_num)
    )

    for client_id in client_ids:
        try:
            # Get client directory name pattern from structure config
            client_prefix = structure.get("client_prefix", "client_")

            # Read results file
            client_output_dir = os.path.join(
                round_dir,
                "clients",
                f"{client_prefix}{client_id}"
            )

            results_file = os.path.join(client_output_dir, "training_results.json")

            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    client_results[client_id] = json.load(f)
            else:
                print(f"Warning: Training results for client {client_id} in round {round_num} not found")
        except Exception as e:
            print(f"Error reading results for client {client_id}: {e}")

    return client_results
