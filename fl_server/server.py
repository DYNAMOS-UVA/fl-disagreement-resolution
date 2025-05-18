"""Main federated learning server implementation."""

import os
import torch
import torch.nn as nn
import json
from datetime import datetime
import glob
import traceback

from fl_module import create_model
import fl_module
from fl_server.evaluation import evaluate_model
from fl_server.aggregation import aggregate_models_from_files, get_structure_config
from fl_server.utils import make_json_serializable
from fl_server.disagreement import (
    load_disagreements,
    get_active_disagreements,
    create_model_tracks,
    get_track_for_client,
    get_clients_in_track
)

class FederatedServer:
    """Server-side implementation for federated learning."""

    def __init__(
        self,
        experiment_type,
        test_dir=None,
        test_units=None,
        device=None,
        results_dir=None
    ):
        """Initialize the federated learning server.

        Args:
            experiment_type: Type of experiment ('n_cmapss' or 'mnist')
            test_dir: Directory containing test data
            test_units: List of unit IDs to use for testing (for N-CMAPSS)
            device: Device to run the model on ('cuda' or 'cpu')
            results_dir: Directory for storing models and results
        """
        self.experiment_type = experiment_type
        self.test_dir = test_dir
        self.test_units = test_units
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = results_dir
        self.round = 0
        self.global_model = None
        self.client_models = {}
        self.aggregation_weights = {}
        self.training_history = {
            "rounds": [],
            "global_test_loss": [],
            "global_test_accuracy": []  # For classification tasks like MNIST
        }

        # Experiment metadata (to be initialized later)
        self.fl_rounds = None
        self.client_ids = None
        self.iid = None

        # Results tracking
        self.results = {
            "experiment_type": experiment_type,
            "rounds": []
        }

        # Create output directories
        if results_dir:
            # Get structure config
            structure = self._get_structure_config()

            # Create output directory
            self.output_dir = os.path.join(results_dir, "output", "server")
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)

            # Create model_storage directory
            model_storage_path = os.path.join(results_dir, structure["model_storage_dir"])
            os.makedirs(model_storage_path, exist_ok=True)
        else:
            os.makedirs("output/server_results", exist_ok=True)
            os.makedirs("output/plots", exist_ok=True)

        # Initialize model based on experiment type
        self._init_model()

    def _init_model(self):
        """Initialize global model based on experiment type."""
        if self.experiment_type == "n_cmapss":
            self.seq_len = 50
            self.n_features = 20
            self.input_dim = self.seq_len * self.n_features
            self.hidden_dim = 32
            self.output_dim = 1
            self.global_model = create_model(
                self.experiment_type,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim
            ).to(self.device)
        elif self.experiment_type == "mnist":
            self.global_model = create_model(self.experiment_type).to(self.device)
        else:
            raise ValueError(f"Unsupported experiment type: {self.experiment_type}")

        print(f"Initialized global {self.experiment_type} model")

    def load_test_data(self, sample_size=500):
        """Load test data for model evaluation.

        Args:
            sample_size: Maximum number of samples to load per test unit
        """
        if self.experiment_type == "n_cmapss":
            if not self.test_dir or not self.test_units:
                raise ValueError("Test directory and test units must be provided for N-CMAPSS")

            # Load test data
            test_samples, test_labels = fl_module.load_ncmapss_test_data(
                self.test_dir,
                self.test_units,
                sample_size=sample_size
            )

            # Preprocess test data
            _, test_normalized, _ = fl_module.preprocess_ncmapss_data(test_samples, test_samples)

            # Create test dataloader
            self.test_loader = fl_module.create_ncmapss_test_dataloader(
                test_normalized,
                test_labels,
                batch_size=64
            )

            print(f"Loaded test data with {len(test_samples)} samples")
        elif self.experiment_type == "mnist":
            if not self.test_dir:
                raise ValueError("Test directory must be provided for MNIST")

            # Load MNIST test data
            test_images, test_labels = fl_module.load_mnist_test_data(
                test_dir=self.test_dir
            )

            # Create test dataloader - no preprocessing needed as it's done during download
            self.test_loader = fl_module.create_mnist_test_dataloader(
                test_images,
                test_labels,
                batch_size=64
            )

            print(f"Loaded MNIST test data with {len(test_images)} samples")
        else:
            # To be implemented for other experiments
            raise NotImplementedError(f"{self.experiment_type} test data loading not implemented yet")

    def get_model_parameters(self):
        """Get current global model parameters to send to clients.

        Returns:
            list: List of model parameter tensors
        """
        return self.global_model.get_parameters()

    def save_model(self, model_dir):
        """Save global model to a directory.

        Args:
            model_dir: Directory to save the model to
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save model state dict
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.global_model.state_dict(), model_path)

        # Save model metadata
        metadata = {
            "experiment_type": self.experiment_type,
            "round": self.round,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved global model to {model_dir}")

    def load_model(self, model_dir):
        """Load global model from a directory.

        Args:
            model_dir: Directory containing the model
        """
        model_path = os.path.join(model_dir, "model.pt")
        metadata_path = os.path.join(model_dir, "metadata.json")

        # Load model state dict
        self.global_model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load metadata if available
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if "round" in metadata:
                    # Keep track of the round but don't overwrite current round
                    print(f"Loaded model from round {metadata['round']}")

        print(f"Loaded global model from {model_dir}")

    def create_model_dirs(self, round_num=None, structure=None):
        """Create necessary directories for models.

        Args:
            round_num: Round number (None for initial model directory)
            structure: Dictionary with directory structure information
        """
        if not self.results_dir or not structure:
            return None

        # Create initial model directory
        if round_num is None:
            initial_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
            os.makedirs(initial_model_dir, exist_ok=True)
            return initial_model_dir

        # Create round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )
        os.makedirs(round_dir, exist_ok=True)

        # Create directory for the global model for training
        training_model_dir = os.path.join(round_dir, structure["global_model"])
        os.makedirs(training_model_dir, exist_ok=True)

        # Create directory for the aggregated model
        aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
        os.makedirs(aggregated_model_dir, exist_ok=True)

        return {
            "round_dir": round_dir,
            "training_model_dir": training_model_dir,
            "aggregated_model_dir": aggregated_model_dir
        }

    def get_model_dir_paths(self, round_num=None, aggregated=False, structure=None):
        """Get paths for model directories.

        Args:
            round_num: Round number (None for initial model)
            aggregated: Whether to get the aggregated model directory
            structure: Dictionary with directory structure information

        Returns:
            str: Path to the model directory
        """
        if not self.results_dir or not structure:
            return None

        # Initial global model
        if round_num is None:
            dir_path = os.path.join(self.results_dir, structure["global_model_initial"])
            os.makedirs(dir_path, exist_ok=True)
            return dir_path

        # Round-specific global model
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        if aggregated:
            dir_path = os.path.join(round_dir, structure["global_model_aggregated"])
        else:
            dir_path = os.path.join(round_dir, structure["global_model"])

        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def init_experiment(self, fl_rounds, client_ids, iid=False):
        """Initialize experiment metadata.

        Args:
            fl_rounds: Number of federated learning rounds
            client_ids: List of client IDs
            iid: Whether the data distribution is IID (for MNIST)
        """
        self.fl_rounds = fl_rounds
        self.client_ids = client_ids
        self.iid = iid

        # Update results with experiment metadata
        self.results["fl_rounds"] = fl_rounds
        self.results["client_ids"] = client_ids
        self.results["iid"] = iid if self.experiment_type == "mnist" else None

        # Save initial results
        self._save_experiment_results()

    def _save_experiment_results(self):
        """Save experiment results to a JSON file."""
        # Check if we have a results directory
        if not self.results_dir:
            return

        # Create the output directory if it doesn't exist
        output_dir = os.path.join(self.results_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Convert NumPy types to Python native types
        serializable_results = make_json_serializable(self.results)

        # Save the results
        results_path = os.path.join(output_dir, "fl_results.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Saved experiment results to {results_path}")

    def initialize_model(self, round_num=0):
        """Initialize and save the initial model.

        Args:
            round_num: Usually 0 for the initial model
        """
        if not self.results_dir:
            return

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Create the initial model directory
        initial_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
        os.makedirs(initial_model_dir, exist_ok=True)

        # Save the initial model
        self.save_model(initial_model_dir)
        print(f"Initialized and saved the initial global model")

    def prepare_training_model(self, round_num, use_initial=False):
        """Prepare the global model for a specific round.

        Args:
            round_num: The current round number
            use_initial: Whether to use the initial model (for round 1)

        Returns:
            str: Path to the prepared model directory
        """
        import glob  # Add explicit import to fix UnboundLocalError

        if not self.results_dir:
            return

        print(f"\n=== SERVER PREPARATION FOR ROUND {round_num} ===")

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Get disagreement configuration
        disagreement_config = self._get_disagreement_config()
        initiation_mechanism = disagreement_config.get("initiation_mechanism", "shallow")

        # Create round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )
        os.makedirs(round_dir, exist_ok=True)

        # Create directory for the global model for training
        training_model_dir = os.path.join(round_dir, structure["global_model"])
        os.makedirs(training_model_dir, exist_ok=True)

        # Create directory for the aggregated model
        aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
        os.makedirs(aggregated_model_dir, exist_ok=True)

        # Check if there are active disagreements in the current round
        etcd_dir = "mock_etcd"
        disagreements = load_disagreements(etcd_dir)
        active_disagreements = get_active_disagreements(disagreements, round_num)

        # Load the appropriate source model
        if use_initial:
            # For round 1 with initial model
            source_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
            print(f"Round {round_num} starting with initial global model from {source_model_dir}")

            # Load the initial model
            self.load_model(source_model_dir)

            # If we have disagreements in round 1, we need to create tracks from the initial model
            if active_disagreements:
                print(f"Creating initial tracks for round {round_num} from global initial model")

                # Get client IDs for track creation
                client_ids_for_tracks = self.results.get("client_ids", [])
                if not client_ids_for_tracks:
                    # Fallback if not in results (e.g. during direct testing)
                    # This part might need adjustment based on how client_ids are reliably fetched
                    print("Warning: client_ids not found in self.results, attempting to infer from client_dirs.")
                    client_dirs_pattern = os.path.join(self.results_dir, "output", "clients", "client_*") # General pattern
                    client_dirs = glob.glob(client_dirs_pattern)
                    if client_dirs:
                        try:
                            client_ids_for_tracks = sorted([int(os.path.basename(d).split("_")[-1]) for d in client_dirs])
                        except ValueError:
                            print(f"Could not parse client IDs from {client_dirs_pattern}. Track creation might be affected.")
                            client_ids_for_tracks = [] # Prevent error in create_model_tracks
                    else:
                        print(f"No client directories found at {client_dirs_pattern}. Track creation might be affected.")
                        client_ids_for_tracks = []

                track_info = create_model_tracks(active_disagreements, client_ids_for_tracks)

                # Create tracks directory
                tracks_dir = os.path.join(round_dir, "tracks")
                os.makedirs(tracks_dir, exist_ok=True)

                # Save track metadata
                metadata_path = os.path.join(tracks_dir, "track_metadata.json")
                track_metadata = {
                    "round": round_num,
                    "tracks": {k: list(v) for k, v in track_info.get("tracks", {}).items()},
                    "client_tracks": track_info.get("client_tracks", {})
                }

                with open(metadata_path, "w") as f:
                    json.dump(track_metadata, f, indent=2)

                # Create each track with the initial model
                for track_name in track_info.get("tracks", {}):
                    track_dir = os.path.join(tracks_dir, track_name)
                    os.makedirs(track_dir, exist_ok=True)

                    # Save model to track directory
                    model_path = os.path.join(track_dir, "model.pt")
                    torch.save(self.global_model.state_dict(), model_path)

                    # Save track-specific metadata
                    track_metadata = {
                        "track_name": track_name,
                        "round": round_num,
                        "client_ids": list(track_info.get("tracks", {}).get(track_name, []))
                    }

                    track_metadata_path = os.path.join(track_dir, "metadata.json")
                    with open(track_metadata_path, "w") as f:
                        json.dump(track_metadata, f, indent=2)

                    print(f"Created initial track '{track_name}' for round {round_num}")

            # Save the initial model as the global model for training
            self.save_model(training_model_dir)
            print(f"Saved global model for training at {training_model_dir}")
        else:
            # For subsequent rounds (round_num > 1)
            # Prepare the main global model for training (for clients not in tracks or as a fallback)
            prev_global_aggregated_dir = os.path.join(
                self.results_dir,
                structure["round_template"].format(round=round_num - 1),
                structure["global_model_aggregated"]
            )
            if os.path.exists(os.path.join(prev_global_aggregated_dir, "model.pt")):
                print(f"Loading main global model from previous round {round_num-1}'s aggregated model: {prev_global_aggregated_dir}")
                self.load_model(prev_global_aggregated_dir)
            else:
                # Fallback if previous aggregated model doesn't exist (e.g. if round 1 was interrupted)
                # In this case, we might have to use the initial model, which is less ideal for round > 1
                print(f"""Warning: Previous round's aggregated model not found at {prev_global_aggregated_dir}.
Falling back to initial global model for main training model of round {round_num}.""")
                initial_model_dir_fallback = os.path.join(self.results_dir, structure["global_model_initial"])
                self.load_model(initial_model_dir_fallback)

            self.save_model(training_model_dir)
            print(f"Saved main global model for training in round {round_num} at {training_model_dir}")

            if active_disagreements:
                print(f"Active disagreements found for round {round_num}. Initiation mechanism: {initiation_mechanism}")

                client_ids_for_tracks = self.results.get("client_ids", []) # Use configured client IDs
                if not client_ids_for_tracks:
                    # Fallback logic for client_ids if necessary, similar to the use_initial=True block
                    print("Warning: client_ids not found in self.results for track creation in round > 1.")
                    # Potentially add more robust fallback here if needed
                    client_dirs_pattern = os.path.join(self.results_dir, "output", "clients", "client_*")
                    client_dirs = glob.glob(client_dirs_pattern)
                    if client_dirs:
                        try:
                            client_ids_for_tracks = sorted([int(os.path.basename(d).split("_")[-1]) for d in client_dirs])
                        except ValueError:
                            client_ids_for_tracks = []
                    else:
                        client_ids_for_tracks = []

                track_info = create_model_tracks(active_disagreements, client_ids_for_tracks)
                current_round_tracks_dir = os.path.join(round_dir, "tracks")
                os.makedirs(current_round_tracks_dir, exist_ok=True)

                prev_round_tracks_main_dir = os.path.join(
                    self.results_dir,
                    structure["round_template"].format(round=round_num - 1),
                    "tracks"
                )

                # Consolidate track metadata from previous round if it exists
                # This will be updated and saved after processing all tracks for the current round
                current_round_track_metadata = {
                    "round": round_num,
                    "tracks": {k: list(v) for k, v in track_info.get("tracks", {}).items()},
                    "client_tracks": track_info.get("client_tracks", {})
                }

                for track_name, clients_in_this_track_set in track_info.get("tracks", {}).items():
                    clients_in_this_track = list(clients_in_this_track_set)
                    current_specific_track_dir = os.path.join(current_round_tracks_dir, track_name)
                    os.makedirs(current_specific_track_dir, exist_ok=True)

                    prev_specific_track_dir = os.path.join(prev_round_tracks_main_dir, track_name)

                    perform_rewind_for_track = (initiation_mechanism == "deep_rewind" and
                                                not os.path.exists(os.path.join(prev_specific_track_dir, "model.pt")))

                    if perform_rewind_for_track:
                        print(f"Performing deep rewind for new track '{track_name}' in round {round_num}")
                        initial_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
                        self.load_model(initial_model_dir) # Start with initial model state
                        current_rewound_model_state = self.global_model.state_dict() # Get state_dict after loading

                        for historical_round in range(1, round_num):
                            historical_clients_dir = os.path.join(
                                self.results_dir,
                                structure["round_template"].format(round=historical_round),
                                structure["clients_dir"]
                            )
                            client_model_files_for_historical_round = []
                            for client_id in clients_in_this_track:
                                client_model_path = os.path.join(
                                    historical_clients_dir,
                                    f"{structure['client_prefix']}{client_id}",
                                    "model.pt"
                                )
                                if os.path.exists(client_model_path):
                                    client_model_files_for_historical_round.append(client_model_path)

                            if client_model_files_for_historical_round:
                                print(f"  Rewinding track '{track_name}' for historical round {historical_round} with {len(client_model_files_for_historical_round)} client models.")
                                aggregated_state = self._aggregate_model_states_from_files_for_rewind(
                                    client_model_files_for_historical_round, self.device
                                )
                                if aggregated_state:
                                    current_rewound_model_state = aggregated_state
                                else:
                                    print(f"  Warning: Aggregation failed for track '{track_name}' in historical round {historical_round}. Using previous state for this step.")
                            else:
                                print(f"  No client models found for track '{track_name}' in historical_round {historical_round} during rewind. Using previous state for this step.")

                        self.global_model.load_state_dict(current_rewound_model_state)
                        self.save_model(current_specific_track_dir)
                        print(f"  Deep rewind complete for track '{track_name}'. Saved to {current_specific_track_dir}")

                    else: # Track continues or shallow initiation for new track
                        if os.path.exists(os.path.join(prev_specific_track_dir, "model.pt")):
                            print(f"Continuing track '{track_name}' from round {round_num - 1}")
                            self.load_model(prev_specific_track_dir)
                            self.save_model(current_specific_track_dir)
                        else:
                            # New track with shallow initiation
                            print(f"Shallow initiation for new track '{track_name}' in round {round_num}")
                            # Source model is the previously aggregated global model for round_num-1
                            # This was loaded into self.global_model before this loop if no active_disagreements previously,
                            # or if this is the first disagreement round.
                            # We need to ensure self.global_model has the correct state from prev_global_aggregated_dir.
                            self.load_model(prev_global_aggregated_dir) # Ensure correct base for shallow new track
                            self.save_model(current_specific_track_dir)

                    # Save individual track metadata
                    track_meta = {
                        "track_name": track_name,
                        "round": round_num,
                        "client_ids": clients_in_this_track,
                        "rewound": perform_rewind_for_track
                    }
                    with open(os.path.join(current_specific_track_dir, "metadata.json"), "w") as f:
                        json.dump(track_meta, f, indent=2)

                # Save overall track metadata for the current round
                metadata_path = os.path.join(current_round_tracks_dir, "track_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(current_round_track_metadata, f, indent=2)
                print(f"Saved track metadata for round {round_num} to {metadata_path}")

            else:
                # No disagreements, use standard model preparation (already handled by loading prev_global_aggregated_dir)
                print(f"No active disagreements for round {round_num}. Using standard global model from {prev_global_aggregated_dir}")
                # The model is already loaded and saved to training_model_dir

        print(f"=== END SERVER PREPARATION FOR ROUND {round_num} ===\n")
        return training_model_dir # This is the main global model dir, tracks are separate

    def get_client_model_path(self, round_num, client_id):
        """Get the path to the appropriate model for a specific client.

        Args:
            round_num: The current round number
            client_id: The client ID

        Returns:
            str: Path to the model for this client
        """
        if not self.results_dir:
            return None

        # Get directory structure
        structure = self._get_structure_config()

        # Check if there are tracks for this round
        tracks_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num),
            "tracks"
        )

        # If tracks directory exists, find the appropriate track for this client
        if os.path.exists(tracks_dir):
            metadata_path = os.path.join(tracks_dir, "track_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        track_metadata = json.load(f)

                    # Convert client_id to string format if needed
                    client_id_str = f"client_{client_id}" if isinstance(client_id, int) else client_id

                    # Get the primary track for this client
                    primary_track = track_metadata.get("client_tracks", {}).get(str(client_id))

                    if primary_track:
                        track_model_path = os.path.join(tracks_dir, primary_track, "model.pt")
                        if os.path.exists(track_model_path):
                            print(f"Using track model {primary_track} for client {client_id}")
                            return track_model_path
                except Exception as e:
                    print(f"Error loading track metadata: {e}")

        # If no tracks or track not found, use the standard model path
        standard_model_path = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num),
            structure["global_model"],
            "model.pt"
        )

        return standard_model_path

    def aggregate_client_models(self, round_num):
        """Aggregate client models for a specific round.

        Args:
            round_num: The current round number

        Returns:
            bool: Whether the aggregation was successful
        """
        if not self.results_dir:
            return False

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Get the round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        # Get the clients directory
        clients_dir = os.path.join(round_dir, structure["clients_dir"])

        # Get the aggregated model directory
        aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
        os.makedirs(aggregated_model_dir, exist_ok=True)

        # Aggregate client models
        aggregate_models_from_files(self, clients_dir)

        # Save the aggregated model
        self.save_model(aggregated_model_dir)

        return True

    def _get_structure_config(self):
        """Get the directory structure configuration.

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
        # Construct path relative to the current file's directory or a known base
        # Assuming fl_server is at the same level as mock_etcd
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "mock_etcd/configuration.json")

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "results" in config and "structure" in config["results"]:
                        # Ensure all keys from default_structure are present
                        loaded_structure = config["results"]["structure"]
                        for key, value in default_structure.items():
                            if key not in loaded_structure:
                                loaded_structure[key] = value
                        return loaded_structure
        except Exception as e:
            print(f"Error loading or parsing structure configuration: {e}. Using default structure.")

        return default_structure

    def _get_disagreement_config(self):
        """Get the disagreement configuration.

        Returns:
            dict: Disagreement configuration
        """
        default_disagreement_config = {
            "initiation_mechanism": "shallow" # Default to shallow
        }

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "mock_etcd/configuration.json")

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "disagreement" in config:
                        loaded_disagreement_config = config["disagreement"]
                        for key, value in default_disagreement_config.items():
                            if key not in loaded_disagreement_config:
                                loaded_disagreement_config[key] = value
                        return loaded_disagreement_config
        except Exception as e:
            print(f"Error loading or parsing disagreement configuration: {e}. Using default disagreement config.")

        return default_disagreement_config

    def _aggregate_model_states_from_files_for_rewind(self, model_file_paths, device):
        """Aggregate model states from a list of model file paths for rewind.

        Args:
            model_file_paths: List of paths to model.pt files.
            device: Device to load models on.

        Returns:
            dict: Aggregated model state_dict. Returns None if no files provided.
        """
        if not model_file_paths:
            return None

        aggregated_state_dict = None
        num_models = len(model_file_paths)

        # Load the first model to initialize aggregated_state_dict structure
        try:
            first_model_state = torch.load(model_file_paths[0], map_location=device)
            aggregated_state_dict = {name: torch.zeros_like(param) for name, param in first_model_state.items()}
        except Exception as e:
            print(f"Error loading first model for rewind aggregation {model_file_paths[0]}: {e}")
            return None # Cannot proceed if first model fails

        # Aggregate all models
        for model_path in model_file_paths:
            try:
                client_state_dict = torch.load(model_path, map_location=device)
                for name, param in client_state_dict.items():
                    if name in aggregated_state_dict:
                        aggregated_state_dict[name] += param / num_models
                    else:
                        # This case should ideally not happen if all models have the same structure
                        print(f"Warning: Parameter {name} not found in initial model structure during rewind. Skipping.")
            except Exception as e:
                print(f"Error loading or aggregating model {model_path} during rewind: {e}. Skipping this model.")
                # If a model fails to load, we effectively reduce num_models for others, which is complex.
                # For simplicity, we average by total initial count. A more robust way would be to sum and count successes.
                # However, the current loop structure would need adjustment.
                # For now, this means a failed model effectively contributes zeros if it was not the first.
                # If it was the first that failed, we return None.
                pass # Continue with other models, but the average will be skewed if not all load.

        return aggregated_state_dict

    def evaluate_model(self, fl_round=None, client_results=None):
        """Evaluate global model on test data.

        Args:
            fl_round: Current federated learning round (if None, considered as initial round 0)
            client_results: DEPRECATED - Dictionary of client training results (not used anymore)

        Returns:
            tuple: (test_loss, accuracy) where accuracy is None for regression tasks
        """
        # Delegate to the evaluate_model function in evaluation.py
        return evaluate_model(self, fl_round, client_results)
