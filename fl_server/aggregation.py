"""Model aggregation functionality for federated learning."""

import os
import torch
import glob
import json

from fl_module import create_model
from fl_server.disagreement import (
    load_disagreements,
    get_active_disagreements,
    create_model_tracks,
    get_track_for_client,
    get_clients_in_track
)

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

    # Load disagreements and create model tracks
    etcd_dir = "mock_etcd"
    disagreements = load_disagreements(etcd_dir)
    active_disagreements = get_active_disagreements(disagreements, server.round)

    # If there are active disagreements, use robust approach with model tracks
    if active_disagreements:
        print(f"Active disagreements found for round {server.round}: {active_disagreements}")
        track_info = create_model_tracks(active_disagreements, client_ids)
        return aggregate_with_tracks(server, clients_dir, track_info, aggregation_weights)
    else:
        # If no disagreements, use standard aggregation
        print(f"No active disagreements for round {server.round}, using standard aggregation")
        return aggregate_standard(server, clients_dir, aggregation_weights)

def aggregate_standard(server, clients_dir, aggregation_weights):
    """Standard model aggregation without tracks.

    Args:
        server: FederatedServer instance
        clients_dir: Directory containing client model directories
        aggregation_weights: Dictionary mapping client IDs to weights

    Returns:
        list: List of aggregated model parameter tensors
    """
    # Find all client directories
    client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
    client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]

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

def aggregate_with_tracks(server, clients_dir, track_info, aggregation_weights):
    """Aggregate models using robust approach with model tracks.

    Args:
        server: FederatedServer instance
        clients_dir: Directory containing client model directories
        track_info: Dictionary containing track information
        aggregation_weights: Dictionary mapping client IDs to weights

    Returns:
        list: List of aggregated model parameter tensors for the first track
    """
    # Find all client directories
    client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
    client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]

    print(f"\n=== AGGREGATION FOR ROUND {server.round} ===")
    print(f"Found {len(client_dirs)} clients: {client_ids}")

    # Initialize temporary model for loading client models
    temp_model = create_model(
        server.experiment_type,
        input_dim=server.input_dim if server.experiment_type == "n_cmapss" else None,
        hidden_dim=server.hidden_dim if server.experiment_type == "n_cmapss" else None,
        output_dim=server.output_dim if server.experiment_type == "n_cmapss" else None
    ).to(server.device)

    # Dictionary to store primary client models
    client_parameters_dict = {}

    # Dictionary to store background client models
    background_parameters_dict = {}

    # Load all client models (both primary and background)
    for client_dir, client_id in zip(client_dirs, client_ids):
        # Load primary client model
        model_path = os.path.join(client_dir, "model.pt")
        if os.path.exists(model_path):
            temp_model.load_state_dict(torch.load(model_path, map_location=server.device))
            # Get client parameters
            client_parameters_dict[client_id] = [p.clone() for p in temp_model.get_parameters()]
            print(f"Loaded primary model from client {client_id}")
        else:
            print(f"Warning: Primary model file not found for client {client_id}")

        # Check for background models
        background_dirs = glob.glob(os.path.join(client_dir, "background_*"))
        for bg_dir in background_dirs:
            bg_track_name = os.path.basename(bg_dir).replace("background_", "", 1)
            bg_model_path = os.path.join(bg_dir, "model.pt")

            if os.path.exists(bg_model_path):
                # Load the background model
                temp_model.load_state_dict(torch.load(bg_model_path, map_location=server.device))

                # Initialize the dictionary for this track if needed
                if bg_track_name not in background_parameters_dict:
                    background_parameters_dict[bg_track_name] = {}

                # Store the background parameters
                background_parameters_dict[bg_track_name][client_id] = [p.clone() for p in temp_model.get_parameters()]
                print(f"Loaded background model from client {client_id} for track {bg_track_name}")

    # Dictionary to store track aggregations
    track_parameters = {}
    tracks = track_info.get("tracks", {})

    print(f"\nAggregating {len(tracks)} tracks:")

    # Count the number of custom tracks (excluding global)
    custom_tracks = [t for t in tracks.keys() if t != "global"]
    has_custom_tracks = len(custom_tracks) > 0

    # Aggregate each track
    for track_name, track_clients in tracks.items():
        # Skip the default track if we have disagreements - we want completely separate tracks
        if track_name == "default" and has_custom_tracks:
            print(f"Skipping default track to ensure track separation in disagreement scenario")
            continue

        print(f"\nAggregating track: '{track_name}' with clients: {sorted(track_clients)}")

        # Skip tracks with no clients
        if not track_clients:
            print(f"Skipping empty track: {track_name}")
            continue

        # Initialize parameters for this track
        track_parameters[track_name] = [torch.zeros_like(param) for param in server.global_model.parameters()]

        # Sum of primary weights for normalization
        primary_weight = 0.0
        primary_clients_aggregated = []

        # First, aggregate primary client models for this track
        for client_id in track_clients:
            # Get client's primary track
            client_primary_track = track_info.get("client_tracks", {}).get(str(client_id))

            # Only include this client's model if this is its primary track
            if client_primary_track == track_name:
                if client_id not in client_parameters_dict:
                    print(f"Warning: Primary parameters not available for client {client_id} in track {track_name}")
                    continue

                # Get client weight (primary clients have full weight)
                weight = aggregation_weights.get(client_id, 1.0 / len(track_clients))
                primary_weight += weight
                primary_clients_aggregated.append(client_id)

                print(f"  Including primary model from client {client_id} with weight {weight}")

                # Add weighted parameters to track parameters
                client_parameters = client_parameters_dict[client_id]
                for i, param in enumerate(client_parameters):
                    track_parameters[track_name][i] += param * weight

        print(f"  Primary models aggregated: {sorted(primary_clients_aggregated)} with total weight {primary_weight:.4f}")

        # Now add background models with lower weight
        # Background clients participate but with reduced weight
        background_clients_aggregated = []
        if track_name in background_parameters_dict:
            background_weight = 0.0

            for client_id, bg_params in background_parameters_dict[track_name].items():
                # Skip client if it's already included as primary
                if track_info.get("client_tracks", {}).get(str(client_id)) == track_name:
                    continue

                # Use a lower weight for background participation (e.g., half the primary weight)
                weight = aggregation_weights.get(client_id, 1.0 / len(track_clients)) * 0.5
                background_weight += weight
                background_clients_aggregated.append(client_id)

                print(f"  Including background model from client {client_id} with reduced weight {weight}")

                # Add weighted parameters
                for i, param in enumerate(bg_params):
                    track_parameters[track_name][i] += param * weight

            print(f"  Background models aggregated: {sorted(background_clients_aggregated)} with total weight {background_weight:.4f}")

            # Adjust normalization to account for background models
            total_weight = primary_weight + background_weight
            if total_weight > 0:
                for i in range(len(track_parameters[track_name])):
                    track_parameters[track_name][i] /= total_weight
                print(f"  Track '{track_name}' normalized with combined weight {total_weight:.4f}")
        elif primary_weight > 0:
            # Normalize primary-only models if weights don't sum to 1
            if abs(primary_weight - 1.0) > 1e-6:
                for i in range(len(track_parameters[track_name])):
                    track_parameters[track_name][i] /= primary_weight
                print(f"  Track '{track_name}' normalized with primary-only weight {primary_weight:.4f}")

        print(f"  Track '{track_name}' aggregation complete.")

    # Save track models to disk
    save_track_models(server, track_parameters, track_info)

    # Update server's global model with parameters from appropriate track
    # For compatibility with server code that expects a global model
    selected_track = None

    # First, try to use the "global" track if it exists
    if "global" in track_parameters:
        selected_track = "global"
    elif track_parameters:
        # Otherwise, use the first available track
        selected_track = next(iter(track_parameters))

    if selected_track:
        server.global_model.set_parameters(track_parameters[selected_track])
        print(f"\nUpdated server's global model with parameters from track: '{selected_track}'")
        print(f"=== END AGGREGATION FOR ROUND {server.round} ===\n")
        return track_parameters[selected_track]
    else:
        print("Warning: No tracks were aggregated")
        print(f"=== END AGGREGATION FOR ROUND {server.round} ===\n")
        # Return the current model parameters if no tracks were aggregated
        return [param.clone() for param in server.global_model.parameters()]

def save_track_models(server, track_parameters, track_info):
    """Save all track models to disk.

    Args:
        server: FederatedServer instance
        track_parameters: Dictionary of track parameters
        track_info: Dictionary of track information
    """
    structure = get_structure_config(server)

    # Check if there are any active disagreements - don't create tracks directory if not
    if not track_info.get("tracks", {}) or (len(track_info.get("tracks", {})) == 1 and "global" in track_info.get("tracks", {})):
        # No active disagreements or only global track, skip creating tracks directory
        print(f"No active disagreements for round {server.round}, skipping track creation")

        # Just update the global model for this round
        global_model_dir = os.path.join(
            server.results_dir,
            structure["round_template"].format(round=server.round),
            structure["global_model_aggregated"]
        )
        os.makedirs(os.path.dirname(global_model_dir), exist_ok=True)

        # Get a reference to a clean model to apply parameters
        temp_model = create_model(
            server.experiment_type,
            input_dim=server.input_dim if server.experiment_type == "n_cmapss" else None,
            hidden_dim=server.hidden_dim if server.experiment_type == "n_cmapss" else None,
            output_dim=server.output_dim if server.experiment_type == "n_cmapss" else None
        ).to(server.device)

        # If we have a "global" track, use that
        if "global" in track_parameters:
            temp_model.set_parameters(track_parameters["global"])
        # Otherwise use the first available track
        elif track_parameters:
            first_track = next(iter(track_parameters))
            temp_model.set_parameters(track_parameters[first_track])

        # Save model
        torch.save(temp_model.state_dict(), global_model_dir)
        print(f"Saved global model for round {server.round}")
        return

    # Create "tracks" directory for this round
    round_dir = os.path.join(
        server.results_dir,
        structure["round_template"].format(round=server.round)
    )
    tracks_dir = os.path.join(round_dir, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)

    # Get a reference to a clean model to apply parameters
    temp_model = create_model(
        server.experiment_type,
        input_dim=server.input_dim if server.experiment_type == "n_cmapss" else None,
        hidden_dim=server.hidden_dim if server.experiment_type == "n_cmapss" else None,
        output_dim=server.output_dim if server.experiment_type == "n_cmapss" else None
    ).to(server.device)

    # Save track metadata
    track_metadata = {
        "round": server.round,
        "tracks": {k: list(v) for k, v in track_info.get("tracks", {}).items()},
        "client_tracks": track_info.get("client_tracks", {})
    }

    metadata_path = os.path.join(tracks_dir, "track_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(track_metadata, f, indent=2)

    # Save each track model
    for track_name, parameters in track_parameters.items():
        track_dir = os.path.join(tracks_dir, track_name)
        os.makedirs(track_dir, exist_ok=True)

        # Apply parameters to temp model
        temp_model.set_parameters(parameters)

        # Save model
        model_path = os.path.join(track_dir, "model.pt")
        torch.save(temp_model.state_dict(), model_path)

        # Save track-specific metadata
        track_specific_metadata = {
            "track_name": track_name,
            "round": server.round,
            "client_ids": list(track_info.get("tracks", {}).get(track_name, []))
        }

        track_metadata_path = os.path.join(track_dir, "metadata.json")
        with open(track_metadata_path, "w") as f:
            json.dump(track_specific_metadata, f, indent=2)

        print(f"Saved track model: {track_name}")

def get_structure_config(server):
    """Get directory structure configuration.

    Args:
        server: FederatedServer instance

    Returns:
        dict: Directory structure configuration
    """
    # Default structure configuration
    default_structure = {
        "model_storage_dir": "model_storage",
        "global_model_initial": "model_storage/global_model_initial",
        "round_template": "model_storage/round_{round}",
        "clients_dir": "clients",
        "global_model": "global_model_for_training",
        "global_model_aggregated": "global_model_aggregated",
        "client_prefix": "client_"
    }

    # Try to load from configuration file
    config_path = os.path.join(os.path.dirname(server.results_dir), "mock_etcd/configuration.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "results" in config and "structure" in config["results"]:
                    return config["results"]["structure"]
    except Exception as e:
        print(f"Error loading configuration: {e}")

    return default_structure
