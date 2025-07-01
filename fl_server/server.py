"""Main federated learning server implementation."""

import os
import torch
import json
import time
from datetime import datetime
import glob

from fl_module import create_model
import fl_module
from fl_server.evaluation import evaluate_model
from fl_server.aggregation import aggregate_models_from_files
from fl_server.utils import make_json_serializable
from fl_server.disagreement import (
    load_disagreements,
    get_active_disagreements,
    create_model_tracks
)

class FederatedServer:
    """Server-side implementation for federated learning."""

    def __init__(
        self,
        experiment_type,
        test_dir=None,
        test_units=None,
        device=None,
        results_dir=None,
        verbose_plots=False
    ):
        """Initialize the federated learning server.

        Args:
            experiment_type: Type of experiment ('n_cmapss' or 'mnist')
            test_dir: Directory containing test data
            test_units: List of unit IDs to use for testing (for N-CMAPSS)
            device: Device to run the model on ('cuda' or 'cpu')
            results_dir: Directory for storing models and results
            verbose_plots: Whether to generate all plots (True) or only minimal plots (False)
        """
        self.experiment_type = experiment_type
        self.test_dir = test_dir
        self.test_units = test_units
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = results_dir
        self.verbose_plots = verbose_plots
        self.round = 0
        self.global_model = None
        self.client_models = {}
        self.aggregation_weights = {}
        self.training_history = {
            "rounds": [],
            "global_test_loss": [],
            "global_test_accuracy": []  # For classification tasks like MNIST
        }

        # Experiment metadata (is initialized in init_experiment)
        self.fl_rounds = None
        self.client_ids = None
        self.iid = None

        # Results tracking
        self.results = {
            "experiment_type": experiment_type,
            "rounds": []
        }

        self.aggregation_timing_history = []
        self.disagreement_settings = self._get_disagreement_config()

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
            "timestamp": datetime.now().isoformat(),
            "disagreement_settings_active_this_round": self.disagreement_settings,
            "fully_excluded_clients_this_round": sorted(list(getattr(self, 'fully_excluded_clients_for_current_round', set())))
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

    def set_total_running_time(self, total_time_seconds):
        """Set the total running time for the federated learning process.

        Args:
            total_time_seconds: Total time in seconds for the complete FL process
        """
        if not hasattr(self, 'aggregation_timing_history'):
            self.aggregation_timing_history = []

        # Add total running time to timing metrics
        self.total_running_time = total_time_seconds

    def _save_experiment_results(self):
        """Save experiment results to a JSON file."""
        # Check if we have a results directory
        if not self.results_dir:
            return

        # Create the output directory if it doesn't exist
        output_dir = os.path.join(self.results_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Add timing metrics to results if available
        if hasattr(self, 'aggregation_timing_history'):
            self.results["aggregation_timing_metrics"] = self.aggregation_timing_history

        # Convert NumPy types to Python native types
        serializable_results = make_json_serializable(self.results)

        # Save the results
        results_path = os.path.join(output_dir, "fl_results.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Saved experiment results to {results_path}")

        # Save timing metrics separately
        if hasattr(self, 'aggregation_timing_history') and self.aggregation_timing_history:
            timing_path = os.path.join(output_dir, "timing_metrics.json")

            # Create timing metrics structure
            timing_data = {
                "total_running_time_seconds": getattr(self, 'total_running_time', None),
                "experiment_init_time_seconds": getattr(self, 'experiment_init_time', None),
                "aggregation_timing_history": self.aggregation_timing_history,
                "round_timing_history": getattr(self, 'round_timing_history', []),
                "evaluation_timing_history": getattr(self, 'evaluation_timing_history', [])
            }

            serializable_timing = make_json_serializable(timing_data)
            with open(timing_path, "w") as f:
                json.dump(serializable_timing, f, indent=2)
            print(f"Saved timing metrics to {timing_path}")

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
        print("Initialized and saved the initial global model")

    def prepare_training_model(self, round_num, use_initial=False):
        """Prepare the global model for a specific round.

        Args:
            round_num: The current round number
            use_initial: Whether to use the initial model (for round 1)

        Returns:
            tuple: (Path to the prepared model directory, preparation time in seconds)
        """

        if not self.results_dir:
            return None, 0.0

        # Timing the track model initialization
        preparation_start_time = time.time()

        print(f"\n=== SERVER PREPARATION FOR ROUND {round_num} ===")
        self.fully_excluded_clients_for_current_round = set()

        structure = self._get_structure_config()
        disagreement_settings = self._get_disagreement_config()
        initiation_mechanism = disagreement_settings.get("initiation_mechanism", "shallow")
        lifting_mechanism = disagreement_settings.get("lifting_mechanism", "shallow")
        finetune_total_rounds = disagreement_settings.get("deep_lifting_finetune_rounds", 3)
        print(f"  Using initiation_mechanism: {initiation_mechanism}, lifting_mechanism: {lifting_mechanism}, deep_lifting_finetune_rounds: {finetune_total_rounds}")

        round_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num))
        os.makedirs(round_dir, exist_ok=True)
        training_model_dir = os.path.join(round_dir, structure["global_model"])
        os.makedirs(training_model_dir, exist_ok=True)

        etcd_dir = "mock_etcd"
        disagreements = load_disagreements(etcd_dir)
        active_disagreements = get_active_disagreements(disagreements, round_num)

        if active_disagreements:
            for client_id_str, disags_list in active_disagreements.items():
                for disag_item in disags_list:
                    if disag_item.get('type') == 'full':
                        try:
                            numeric_id = int(client_id_str.split('_')[-1]) if '_' in client_id_str else int(client_id_str)
                            self.fully_excluded_clients_for_current_round.add(numeric_id)
                        except ValueError:
                            print(f"Warning: Could not parse numeric ID from client_id_str '{client_id_str}' for full exclusion.")
            if self.fully_excluded_clients_for_current_round:
                print(f"  Fully excluded clients identified for round {round_num}: {sorted(list(self.fully_excluded_clients_for_current_round))}")

        if use_initial:
            source_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
            print(f"Round {round_num} starting with initial global model from {source_model_dir}")
            self.load_model(source_model_dir)

            if active_disagreements:
                print(f"Creating initial tracks for round {round_num} from global initial model")
                client_ids_for_tracks = self.results.get("client_ids", [])
                if not client_ids_for_tracks:
                    print("Warning: client_ids not found in self.results, attempting to infer from client_dirs.")
                    client_dirs_pattern = os.path.join(self.results_dir, "output", "clients", "client_*")
                    client_dirs = glob.glob(client_dirs_pattern)
                    client_ids_for_tracks = sorted([int(os.path.basename(d).split("_")[-1]) for d in client_dirs]) if client_dirs else []
                    if not client_ids_for_tracks:
                         print(f"No client directories found at {client_dirs_pattern}. Track creation might be affected.")

                track_info = create_model_tracks(active_disagreements, client_ids_for_tracks)
                tracks_dir = os.path.join(round_dir, "tracks")
                os.makedirs(tracks_dir, exist_ok=True)
                metadata_path = os.path.join(tracks_dir, "track_metadata.json")
                track_metadata_content = {
                    "round": round_num,
                    "tracks": {k: list(v) for k, v in track_info.get("tracks", {}).items()},
                    "client_tracks": track_info.get("client_tracks", {})
                }
                with open(metadata_path, "w") as f:
                    json.dump(track_metadata_content, f, indent=2)

                for track_name_iter in track_info.get("tracks", {}):
                    track_dir_iter = os.path.join(tracks_dir, track_name_iter)
                    os.makedirs(track_dir_iter, exist_ok=True)
                    torch.save(self.global_model.state_dict(), os.path.join(track_dir_iter, "model.pt"))
                    individual_track_meta = {
                        "track_name": track_name_iter,
                        "round": round_num,
                        "client_ids": list(track_info.get("tracks", {}).get(track_name_iter, [])),
                        "rewound_this_round": False, # Cannot be rewound in round 1 from initial
                        "finetuning_status": {}
                    }
                    with open(os.path.join(track_dir_iter, "metadata.json"), "w") as f_meta_track:
                        json.dump(individual_track_meta, f_meta_track, indent=2)
                    print(f"Created initial track '{track_name_iter}' for round {round_num}")

            self.save_model(training_model_dir)
            print(f"Saved global model for training at {training_model_dir}")
        else: # Not use_initial (round_num > 1 typically)
            prev_global_aggregated_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num - 1), structure["global_model_aggregated"])
            if os.path.exists(os.path.join(prev_global_aggregated_dir, "model.pt")):
                print(f"Loading main global model from previous round {round_num-1}'s aggregated model: {prev_global_aggregated_dir}")
                self.load_model(prev_global_aggregated_dir)
            else:
                print(f"""Warning: Previous round's aggregated model not found at {prev_global_aggregated_dir}. Falling back to initial global model for main training model of round {round_num}.""")
                initial_model_dir_fallback = os.path.join(self.results_dir, structure["global_model_initial"])
                self.load_model(initial_model_dir_fallback)
            self.save_model(training_model_dir)
            print(f"Saved main global model for training in round {round_num} at {training_model_dir}")

            if active_disagreements:
                print(f"Active disagreements found for round {round_num}. Initiation mechanism: {initiation_mechanism}")
                client_ids_for_tracks = self.results.get("client_ids", [])
                if not client_ids_for_tracks: # Fallback for client_ids
                    print("Warning: client_ids not found in self.results for track creation in round > 1.")
                    client_dirs_pattern = os.path.join(self.results_dir, "output", "clients", "client_*")
                    client_dirs = glob.glob(client_dirs_pattern)
                    client_ids_for_tracks = sorted([int(os.path.basename(d).split("_")[-1]) for d in client_dirs]) if client_dirs else []

                track_info = create_model_tracks(active_disagreements, client_ids_for_tracks)
                current_round_tracks_dir = os.path.join(round_dir, "tracks")
                os.makedirs(current_round_tracks_dir, exist_ok=True)
                prev_round_tracks_main_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num - 1), "tracks")

                current_round_track_metadata_content = {
                    "round": round_num,
                    "tracks": {k: list(v) for k, v in track_info.get("tracks", {}).items()},
                    "client_tracks": track_info.get("client_tracks", {})
                }

                for track_name, clients_in_this_track_set in track_info.get("tracks", {}).items():
                    clients_in_this_track_list = list(clients_in_this_track_set)
                    current_specific_track_dir = os.path.join(current_round_tracks_dir, track_name)
                    os.makedirs(current_specific_track_dir, exist_ok=True)
                    prev_specific_track_dir = os.path.join(prev_round_tracks_main_dir, track_name)
                    track_existed_previously_as_specific_dir = os.path.exists(os.path.join(prev_specific_track_dir, "model.pt"))

                    current_track_finetuning_status = {}
                    if lifting_mechanism == "deep_incr_finetune":
                        print(f"    Deep incremental finetuning analysis for track '{track_name}':")
                        print(f"      Total finetuning rounds: {finetune_total_rounds}")
                        print(f"      Clients in track: {sorted(clients_in_this_track_list)}")

                        prev_finetune_status_path = os.path.join(prev_specific_track_dir, "finetuning_status.json")
                        prev_track_finetuning_status_loaded = {}
                        if os.path.exists(prev_finetune_status_path):
                            try:
                                with open(prev_finetune_status_path, 'r') as f_fs:
                                    prev_track_finetuning_status_loaded = json.load(f_fs)
                                print(f"      Previous finetuning status: {prev_track_finetuning_status_loaded}")
                            except Exception as e:
                                print(f"      Warning: Could not load previous finetuning status: {e}")
                        else:
                            print("      No previous finetuning status file found")

                        prev_track_clients_metadata = set()
                        prev_track_metadata_path_iter = os.path.join(prev_specific_track_dir, "metadata.json")
                        if os.path.exists(prev_track_metadata_path_iter):
                            try:
                                with open(prev_track_metadata_path_iter, 'r') as f_meta:
                                    prev_track_clients_metadata = set(json.load(f_meta).get("client_ids", []))
                                print(f"      Previous track clients: {sorted(list(prev_track_clients_metadata))}")
                            except Exception as e:
                                print(f"      Warning: Could not load previous metadata: {e}")
                        else:
                            print("      No previous track metadata found")

                        # Analyze client finetuning needs
                        clients_new_to_track = []
                        clients_continuing_ft = []
                        clients_completed_ft = []
                        clients_no_ft = []

                        print("      Analyzing finetuning for each client:")
                        for client_id_numeric in clients_in_this_track_list:
                            client_id_str_iter = str(client_id_numeric)
                            if client_id_numeric not in prev_track_clients_metadata and track_existed_previously_as_specific_dir:
                                print(f"        Client {client_id_str_iter}: New to track → Starting finetuning (1/{finetune_total_rounds})")
                                current_track_finetuning_status[client_id_str_iter] = 1
                                clients_new_to_track.append(client_id_str_iter)
                            elif client_id_str_iter in prev_track_finetuning_status_loaded:
                                progress = prev_track_finetuning_status_loaded[client_id_str_iter] + 1
                                if progress <= finetune_total_rounds:
                                    print(f"        Client {client_id_str_iter}: Continuing finetuning → Round {progress}/{finetune_total_rounds}")
                                    current_track_finetuning_status[client_id_str_iter] = progress
                                    clients_continuing_ft.append(f"{client_id_str_iter}({progress}/{finetune_total_rounds})")
                                else:
                                    print(f"        Client {client_id_str_iter}: Completed finetuning → No further action")
                                    clients_completed_ft.append(client_id_str_iter)
                            else:
                                print(f"        Client {client_id_str_iter}: No finetuning required")
                                clients_no_ft.append(client_id_str_iter)

                        # Summary of finetuning actions for this track
                        print(f"      Track '{track_name}' finetuning summary:")
                        if clients_new_to_track:
                            print(f"        New to track: {clients_new_to_track}")
                        if clients_continuing_ft:
                            print(f"        Continuing: {clients_continuing_ft}")
                        if clients_completed_ft:
                            print(f"        Completed: {clients_completed_ft}")
                        if clients_no_ft:
                            print(f"        No action: {clients_no_ft}")

                    if current_track_finetuning_status:
                        with open(os.path.join(current_specific_track_dir, "finetuning_status.json"), 'w') as f_fs_curr:
                            json.dump(current_track_finetuning_status, f_fs_curr, indent=2)
                        print(f"    Saved finetuning status for track '{track_name}': {current_track_finetuning_status}")
                    else:
                        print(f"    No clients require finetuning in track '{track_name}'")

                    print(f"  Evaluating track: '{track_name}'. Existed previously: {track_existed_previously_as_specific_dir}")
                    is_new_track_to_dir_structure = not track_existed_previously_as_specific_dir
                    composition_has_changed = False
                    if is_new_track_to_dir_structure:
                        if track_name == "global":
                            all_system_clients = set(self.results.get("client_ids", []))
                            active_disags_prev_round = get_active_disagreements(disagreements, round_num - 1 if round_num > 0 else 0)
                            fully_excluded_prev = set()
                            if active_disags_prev_round:
                                for cid_str, d_list in active_disags_prev_round.items():
                                    for d_item in d_list:
                                        if d_item.get('type') == 'full':
                                            try:
                                                fully_excluded_prev.add(int(cid_str.split('_')[-1]) if '_' in cid_str else int(cid_str))
                                            except ValueError:
                                                print(f"Warning: Could not parse client ID '{cid_str}' during rewind check for global track.")
                            conceptual_prev_global_clients = all_system_clients - fully_excluded_prev
                            if clients_in_this_track_set != conceptual_prev_global_clients:
                                composition_has_changed = True
                                print("    Global track is new to 'tracks' dir. Composition changed.")
                            else:
                                print("    Global track is new to 'tracks' dir. Composition UNCHANGED.")
                        else: # Non-global track, new to dir structure
                            composition_has_changed = True
                            print(f"    Non-global track '{track_name}' is new. Marking composition changed.")
                    else: # Track existed previously
                        prev_meta_path = os.path.join(prev_specific_track_dir, "metadata.json")
                        if os.path.exists(prev_meta_path):
                            with open(prev_meta_path, 'r') as f_m:
                                prev_clients = set(json.load(f_m).get("client_ids", []))
                            if prev_clients != clients_in_this_track_set:
                                composition_has_changed = True
                                print(f"    Track '{track_name}' existed and composition changed.")
                            else:
                                print(f"    Track '{track_name}' existed and composition UNCHANGED.")
                        else:
                            composition_has_changed = True
                            print(f"    Track '{track_name}' existed but no prev metadata. Marking changed.")

                    perform_rewind_for_this_track = (initiation_mechanism == "deep_rewind" and composition_has_changed)
                    print(f"    Perform rewind for '{track_name}': {perform_rewind_for_this_track}")

                    if perform_rewind_for_this_track:
                        print(f"    Performing deep rewind for track '{track_name}':")
                        self.load_model(os.path.join(self.results_dir, structure["global_model_initial"]))
                        current_rewound_model_state = self.global_model.state_dict()
                        for hist_round in range(1, round_num):
                            hist_clients_dir = os.path.join(self.results_dir, structure["round_template"].format(round=hist_round), structure["clients_dir"])
                            client_model_files_hist = [os.path.join(hist_clients_dir, f"{structure['client_prefix']}{cid}", "model.pt") for cid in clients_in_this_track_list if os.path.exists(os.path.join(hist_clients_dir, f"{structure['client_prefix']}{cid}", "model.pt"))]
                            if client_model_files_hist:
                                # Extract client IDs from file paths for better logging
                                client_ids_in_round = []
                                for file_path in client_model_files_hist:
                                    client_dir = os.path.basename(os.path.dirname(file_path))
                                    client_id = client_dir.replace(structure['client_prefix'], "")
                                    client_ids_in_round.append(client_id)
                                client_ids_str = ", ".join(sorted(client_ids_in_round))
                                print(f"      Round {hist_round}: Aggregating {len(client_model_files_hist)} models from clients [{client_ids_str}]")
                                aggregated_state = self._aggregate_model_states_from_files_for_rewind(client_model_files_hist, self.device)
                                if aggregated_state:
                                    current_rewound_model_state = aggregated_state
                                else:
                                    print(f"        Warning: Aggregation failed for '{track_name}' in round {hist_round}.")
                            else:
                                print(f"      Round {hist_round}: No models available for '{track_name}' in rewind.")
                        self.global_model.load_state_dict(current_rewound_model_state)
                        self.save_model(current_specific_track_dir)
                        print(f"    Deep rewind complete for '{track_name}'. Saved to {current_specific_track_dir}")
                    else: # Not rewinding this track
                        source_model_for_track_path = prev_specific_track_dir if track_existed_previously_as_specific_dir else prev_global_aggregated_dir
                        print(f"Loading model for track '{track_name}' from '{source_model_for_track_path}'.")
                        if os.path.exists(os.path.join(source_model_for_track_path, "model.pt")):
                            self.load_model(source_model_for_track_path)
                        else:
                            print(f"Warning: Source model '{source_model_for_track_path}' for track '{track_name}' not found. Using initial global model.")
                            self.load_model(os.path.join(self.results_dir, structure["global_model_initial"]))
                        self.save_model(current_specific_track_dir)

                    individual_track_meta = {"track_name": track_name, "round": round_num, "client_ids": clients_in_this_track_list, "rewound_this_round": perform_rewind_for_this_track,
                                           "finetuning_status": {cid_str: f"{prog}/{finetune_total_rounds}" for cid_str, prog in current_track_finetuning_status.items()} if current_track_finetuning_status else {}}
                    with open(os.path.join(current_specific_track_dir, "metadata.json"), "w") as f_track_meta:
                        json.dump(individual_track_meta, f_track_meta, indent=2)

                with open(os.path.join(current_round_tracks_dir, "track_metadata.json"), "w") as f_meta_overall:
                    json.dump(current_round_track_metadata_content, f_meta_overall, indent=2)
                print(f"Saved track metadata for round {round_num}.")
            else: # No active disagreements
                print(f"No active disagreements for round {round_num}. Using standard global model from {prev_global_aggregated_dir}")
                if round_num > 1 and lifting_mechanism == "deep_incr_finetune":
                    print(f"    Deep incremental finetuning check for round {round_num}:")
                    print(f"      - Mechanism: {lifting_mechanism}")
                    print(f"      - Total finetuning rounds: {finetune_total_rounds}")
                    print("      - No active tracks detected")
                    current_global_finetuning_status = {}
                    prev_round_main_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num - 1))
                    prev_global_finetune_status_path = os.path.join(prev_round_main_dir, "global_finetuning_status.json")
                    prev_global_finetuning_status_loaded = {}
                    if os.path.exists(prev_global_finetune_status_path):
                        try:
                            with open(prev_global_finetune_status_path, 'r') as f_fs:
                                prev_global_finetuning_status_loaded = json.load(f_fs)
                            print(f"      Loaded previous finetuning status from R{round_num-1}: {prev_global_finetuning_status_loaded}")
                        except Exception as e:
                            print(f"      Warning: Could not load previous global finetuning status: {e}")

                    prev_round_track_metadata_path = os.path.join(prev_round_main_dir, "tracks", "track_metadata.json")
                    prev_round_had_active_tracks = os.path.exists(prev_round_track_metadata_path)
                    prev_round_client_track_map = {} # Stores client_id_str -> track_name from previous round

                    if prev_round_had_active_tracks:
                        print(f"      Tracks were active in previous round (R{round_num-1}). Checking for clients rejoining from non-global tracks.")
                        try:
                            with open(prev_round_track_metadata_path, 'r') as f_prev_meta:
                                prev_track_meta_content = json.load(f_prev_meta)
                                # Ensure keys are strings for consistent lookup
                                prev_round_client_track_map = {str(k): v for k, v in prev_track_meta_content.get("client_tracks", {}).items()}
                        except Exception as e:
                            print(f"        Warning: Could not load client_tracks from previous round's track_metadata.json: {e}")
                    else:
                        print(f"      No tracks were active in previous round (R{round_num-1}). Finetuning initiation based on client absence or status.")

                    prev_round_clients_dir = os.path.join(prev_round_main_dir, structure["clients_dir"])
                    prev_round_submitted_model_ids = set()
                    if os.path.exists(prev_round_clients_dir):
                        client_model_dirs = glob.glob(os.path.join(prev_round_clients_dir, f"{structure['client_prefix']}*"))
                        for d_path in client_model_dirs:
                            try:
                                prev_round_submitted_model_ids.add(int(os.path.basename(d_path).replace(structure['client_prefix'], "")))
                            except ValueError:
                                pass
                    print(f"      Previous round (R{round_num-1}) participants: {sorted(list(prev_round_submitted_model_ids))}")
                    current_global_participants = set(self.results.get("client_ids", [])) - self.fully_excluded_clients_for_current_round
                    print(f"      Current round (R{round_num}) participants: {sorted(list(current_global_participants))}")

                    # Analyze client changes
                    newly_joining = current_global_participants - prev_round_submitted_model_ids
                    continuing = current_global_participants & prev_round_submitted_model_ids
                    if newly_joining:
                        print(f"      Newly joining clients: {sorted(list(newly_joining))}")
                    if continuing:
                        print(f"      Continuing clients: {sorted(list(continuing))}")

                    print("      Analyzing finetuning requirements for each client:")

                    clients_starting_new = []
                    clients_continuing = []
                    clients_completed = []
                    clients_no_action = []

                    for client_id_numeric in current_global_participants:
                        client_id_str_gf = str(client_id_numeric)
                        start_new_finetuning_for_client = False
                        if client_id_numeric not in prev_round_submitted_model_ids:
                            print(f"        Client {client_id_str_gf}: Was absent in R{round_num-1}, now joining → Starting finetuning")
                            start_new_finetuning_for_client = True
                        elif prev_round_had_active_tracks:
                            client_prev_track = prev_round_client_track_map.get(client_id_str_gf)
                            if client_prev_track and client_prev_track != "global":
                                print(f"        Client {client_id_str_gf}: Rejoining from track '{client_prev_track}' → Starting finetuning")
                                start_new_finetuning_for_client = True
                            else:
                                # Client was present, tracks existed, but client was on 'global' track or track info missing for them.
                                # No NEW finetuning initiation due to track dissolution itself.
                                print(f"        Client {client_id_str_gf}: Was on '{client_prev_track or 'global'}' track → No new finetuning from track dissolution")

                        if start_new_finetuning_for_client:
                            current_global_finetuning_status[client_id_str_gf] = 1
                            clients_starting_new.append(client_id_str_gf)
                        elif client_id_str_gf in prev_global_finetuning_status_loaded:
                            # Not starting NEW, but might be CONTINUING a previous global finetune cycle
                            progress_gf = prev_global_finetuning_status_loaded[client_id_str_gf] + 1
                            if progress_gf <= finetune_total_rounds:
                                print(f"        Client {client_id_str_gf}: Continuing finetuning → Round {progress_gf}/{finetune_total_rounds}")
                                current_global_finetuning_status[client_id_str_gf] = progress_gf
                                clients_continuing.append(f"{client_id_str_gf}({progress_gf}/{finetune_total_rounds})")
                            else:
                                print(f"        Client {client_id_str_gf}: Completed finetuning → No further action needed")
                                clients_completed.append(client_id_str_gf)
                        else:
                            print(f"        Client {client_id_str_gf}: No finetuning action required")
                            clients_no_action.append(client_id_str_gf)
                        # Else: Client was present, no new trigger, and not in prev_global_finetuning_status_loaded -> no finetuning action for this client.

                    print("      Finetuning summary:")
                    if clients_starting_new:
                        print(f"        Starting new: {clients_starting_new}")
                    if clients_continuing:
                        print(f"        Continuing: {clients_continuing}")
                    if clients_completed:
                        print(f"        Completed: {clients_completed}")
                    if clients_no_action:
                        print(f"        No action: {clients_no_action}")

                    if current_global_finetuning_status:
                        current_global_finetune_status_path = os.path.join(round_dir, "global_finetuning_status.json")
                        try:
                            with open(current_global_finetune_status_path, 'w') as f_fs_global:
                                json.dump(current_global_finetuning_status, f_fs_global, indent=2)
                            print(f"      Saved global finetuning status: {current_global_finetuning_status}")
                        except Exception as e:
                            print(f"      Warning: Could not save global finetuning status: {e}")
                    else:
                        print("      No clients require finetuning this round")

        print(f"=== END SERVER PREPARATION FOR ROUND {round_num} ===\\n")

        # Timing the track model initialization
        preparation_time = time.time() - preparation_start_time
        print(f"Track model initialization completed in {preparation_time:.4f} seconds")

        return training_model_dir, preparation_time

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

    def aggregate_with_disagreement_resolution(self, round_num):
        """Aggregate client models using disagreement-aware track-based algorithm.

        This method performs sophisticated model aggregation that handles disagreements
        by creating separate model tracks and aggregating primary/background models
        according to the disagreement resolution strategy.

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

        clients_dir = os.path.join(round_dir, structure["clients_dir"])

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
            "initiation_mechanism": "shallow", # Default to shallow
            "lifting_mechanism": "shallow",    # Default to shallow
            "deep_lifting_finetune_rounds": 3  # Default to 3 rounds
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
                pass # Continue with other models

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
