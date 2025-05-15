"""Model aggregation functionality for federated learning."""

import os
import torch
import glob

from fl_module import create_model

def aggregate_models_from_files(server, clients_dir, aggregation_weights=None):
    """Aggregate models from client files.

    Args:
        server: FederatedServer instance
        clients_dir: Directory containing client model directories
        aggregation_weights: Optional dictionary mapping client IDs to weights

    Returns:
        list: List of aggregated model parameter tensors
    """
    server.round += 1
    print(f"Starting file-based aggregation for round {server.round}")

    # Find all client directories
    client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
    client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]

    print(f"Found {len(client_dirs)} clients: {client_ids}")

    # If no weights provided, use equal weighting
    if aggregation_weights is None:
        n_clients = len(client_dirs)
        aggregation_weights = {client_id: 1.0 / n_clients for client_id in client_ids}

    # Initialize temporary model for loading client models
    temp_model = create_model(
        server.experiment_type,
        input_dim=server.input_dim if server.experiment_type == "n_cmapss" else None,
        hidden_dim=server.hidden_dim if server.experiment_type == "n_cmapss" else None,
        output_dim=server.output_dim if server.experiment_type == "n_cmapss" else None
    ).to(server.device)

    # Initialize new global parameters with zeros
    global_parameters = [torch.zeros_like(param) for param in server.global_model.parameters()]

    # Load each client model and aggregate parameters
    for client_dir, client_id in zip(client_dirs, client_ids):
        # Load client model
        model_path = os.path.join(client_dir, "model.pt")
        temp_model.load_state_dict(torch.load(model_path, map_location=server.device))

        # Get client parameters
        client_parameters = temp_model.get_parameters()

        # Get client weight
        weight = aggregation_weights.get(client_id, 1.0 / len(client_dirs))

        # Add weighted parameters to global parameters
        for i, param in enumerate(client_parameters):
            global_parameters[i] += param * weight

    # Update global model with aggregated parameters
    server.global_model.set_parameters(global_parameters)
    print(f"Updated global model with parameters from {len(client_dirs)} clients")

    return global_parameters
