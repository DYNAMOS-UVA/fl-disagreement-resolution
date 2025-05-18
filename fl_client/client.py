"""Main federated learning client implementation."""

import os
import torch
import json
from datetime import datetime

from fl_module.models import create_model
import fl_module
from fl_client.utils import save_training_results
from fl_client.training import train_model

class FederatedClient:
    """Client-side implementation for federated learning."""

    def __init__(
        self,
        client_id,
        experiment_type,
        data_dir,
        batch_size=64,
        epochs=5,
        learning_rate=0.001,
        device=None,
        results_dir=None
    ):
        """Initialize the federated learning client.

        Args:
            client_id: Client ID
            experiment_type: Type of experiment ('n_cmapss' or 'mnist')
            data_dir: Directory containing client data
            batch_size: Batch size for training
            epochs: Number of local training epochs
            learning_rate: Learning rate for optimization
            device: Device to run the model on ('cuda' or 'cpu')
            results_dir: Directory for storing models and results
        """
        self.client_id = client_id
        self.experiment_type = experiment_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = results_dir

        # Initialize model and training parameters based on experiment type
        if experiment_type == "n_cmapss":
            self.seq_len = 50
            self.n_features = 20
            self.input_dim = self.seq_len * self.n_features
            self.hidden_dim = 32
            self.output_dim = 1
            self.model = create_model(
                experiment_type,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim
            ).to(self.device)
        elif experiment_type == "mnist":
            self.model = create_model(experiment_type).to(self.device)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

        # Create results directory
        if results_dir:
            self.output_dir = os.path.join(results_dir, "output", "clients", f"client_{client_id}")
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            os.makedirs("output/client_results", exist_ok=True)

    def create_model_dir(self, round_num, structure=None):
        """Create client model directory for a specific round.

        Args:
            round_num: The round number
            structure: Dictionary with directory structure information

        Returns:
            str: Path to the client model directory
        """
        if not self.results_dir or not structure:
            return None

        # Create the client directory for this round
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        clients_dir = os.path.join(round_dir, structure["clients_dir"])
        os.makedirs(clients_dir, exist_ok=True)

        client_dir = os.path.join(clients_dir, f"{structure['client_prefix']}{self.client_id}")
        os.makedirs(client_dir, exist_ok=True)

        return client_dir

    def load_data(self, sample_size=1000):
        """Load and preprocess client data.

        Args:
            sample_size: Maximum number of samples to load
        """
        if self.experiment_type == "n_cmapss":
            # Load data for this client
            samples, labels = fl_module.load_ncmapss_client_data(
                self.client_id,
                self.data_dir,
                sample_size=sample_size
            )

            # Preprocess data
            samples_normalized, _ = fl_module.preprocess_ncmapss_data(samples)

            # Create dataloaders
            self.train_loader, self.valid_loader = fl_module.create_ncmapss_client_dataloaders(
                samples_normalized,
                labels,
                batch_size=self.batch_size
            )

            print(f"Client {self.client_id} loaded {len(samples)} samples")
        elif self.experiment_type == "mnist":
            # Load MNIST data for this client
            images, labels = fl_module.load_mnist_client_data(
                self.client_id,
                train_dir=self.data_dir,
                sample_size=sample_size
            )

            # Create dataloaders
            self.train_loader, self.valid_loader = fl_module.create_mnist_client_dataloaders(
                images,
                labels,
                batch_size=self.batch_size
            )

            print(f"Client {self.client_id} loaded {len(images)} MNIST samples")
        else:
            # For other experiments
            raise NotImplementedError(f"{self.experiment_type} data loading not implemented yet")

    def save_model(self, model_dir):
        """Save model to a directory.

        Args:
            model_dir: Directory to save the model to
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save model state dict
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save model metadata
        metadata = {
            "client_id": self.client_id,
            "experiment_type": self.experiment_type,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Client {self.client_id} saved model to {model_dir}")

    def load_model(self, model_dir):
        """Load model from a directory.

        Args:
            model_dir: Directory containing the model
        """
        model_path = os.path.join(model_dir, "model.pt")

        # Load model state dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Client {self.client_id} loaded model from {model_dir}")

    def load_round_model(self, round_num):
        """Load the global model for a specific round.

        Args:
            round_num: The current round number

        Returns:
            bool: Whether the model was successfully loaded
        """
        if not self.results_dir:
            return False

        print(f"\n=== CLIENT {self.client_id} MODEL LOADING FOR ROUND {round_num} ===")

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Get the round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        # Check if there are tracks for this round
        tracks_dir = os.path.join(round_dir, "tracks")
        primary_track_loaded = False

        # Clear any existing background tracks
        self.background_tracks = []

        if os.path.exists(tracks_dir):
            print(f"Found tracks directory at: {tracks_dir}")

            # Look for track metadata
            metadata_path = os.path.join(tracks_dir, "track_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        track_metadata = json.load(f)

                    track_names = list(track_metadata.get("tracks", {}).keys())
                    print(f"Found {len(track_names)} tracks: {track_names}")

                    # Find this client's primary track
                    primary_track = track_metadata.get("client_tracks", {}).get(str(self.client_id))

                    if primary_track:
                        print(f"Client {self.client_id} is assigned to primary track: '{primary_track}'")

                        # Look for track metadata to get details about this track
                        track_metadata_path = os.path.join(tracks_dir, primary_track, "metadata.json")
                        if os.path.exists(track_metadata_path):
                            try:
                                with open(track_metadata_path, 'r') as f:
                                    primary_track_metadata = json.load(f)
                                print(f"Primary track '{primary_track}' info: {primary_track_metadata}")
                            except Exception as e:
                                print(f"Error reading track metadata: {e}")

                        # Load the model from this track
                        track_dir = os.path.join(tracks_dir, primary_track)
                        if os.path.exists(track_dir):
                            model_path = os.path.join(track_dir, "model.pt")
                            if os.path.exists(model_path):
                                # Load the model
                                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                                print(f"Successfully loaded primary track model from {track_dir}")

                                # Check if this is a continuation from a previous round's track
                                if os.path.exists(track_metadata_path):
                                    previous_round = primary_track_metadata.get("previous_round")
                                    if previous_round is not None:
                                        print(f"This model continues from the same track in round {previous_round}")
                                    else:
                                        print(f"This is a new track model created for round {round_num}")

                                primary_track_loaded = True

                                # Now check for background tracks this client should train on
                                participation_tracks = []
                                for track_name, track_clients in track_metadata.get("tracks", {}).items():
                                    # Convert track_clients to integers for comparison if they're strings
                                    track_clients_int = [int(c) if isinstance(c, str) else c for c in track_clients]
                                    if track_name != primary_track and self.client_id in track_clients_int:
                                        participation_tracks.append(track_name)

                                if participation_tracks:
                                    print(f"Client {self.client_id} will also train on background tracks: {participation_tracks}")
                                    self.background_tracks = []

                                    for bg_track in participation_tracks:
                                        bg_track_dir = os.path.join(tracks_dir, bg_track)
                                        if os.path.exists(bg_track_dir):
                                            bg_model_path = os.path.join(bg_track_dir, "model.pt")
                                            if os.path.exists(bg_model_path):
                                                # Create a separate model for this background track
                                                if self.experiment_type == "n_cmapss":
                                                    bg_model = create_model(
                                                        self.experiment_type,
                                                        input_dim=self.input_dim,
                                                        hidden_dim=self.hidden_dim,
                                                        output_dim=self.output_dim
                                                    ).to(self.device)
                                                else:
                                                    bg_model = create_model(self.experiment_type).to(self.device)

                                                # Load the model weights
                                                bg_model.load_state_dict(torch.load(bg_model_path, map_location=self.device))
                                                self.background_tracks.append({
                                                    "name": bg_track,
                                                    "model": bg_model,
                                                    "dir": bg_track_dir
                                                })
                                                print(f"Successfully loaded background track model '{bg_track}' from {bg_track_dir}")
                                else:
                                    print(f"Client {self.client_id} has no background tracks to train on")
                            else:
                                print(f"Warning: Model file not found at {model_path}")
                        else:
                            print(f"Warning: Track directory not found at {track_dir}")
                    else:
                        print(f"Client {self.client_id} has no assigned primary track, will use global model")
                except Exception as e:
                    print(f"Error loading track models: {e}")
            else:
                print(f"No track metadata found at {metadata_path}")
        else:
            print(f"No tracks directory found at {tracks_dir}, will use standard global model")

        # If no tracks found or error occurred, load the standard global model
        if not primary_track_loaded:
            global_model_dir = os.path.join(round_dir, structure["global_model"])
            if os.path.exists(global_model_dir):
                model_path = os.path.join(global_model_dir, "model.pt")
                if os.path.exists(model_path):
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Successfully loaded standard global model from {global_model_dir}")
                    self.background_tracks = []  # No background tracks in standard mode
                    print(f"=== END CLIENT {self.client_id} MODEL LOADING ===\n")
                    return True
                else:
                    print(f"Warning: Model file not found at {model_path}")
            else:
                print(f"Warning: Global model directory not found at {global_model_dir}")

            print(f"=== END CLIENT {self.client_id} MODEL LOADING ===\n")
            return False

        print(f"=== END CLIENT {self.client_id} MODEL LOADING ===\n")
        return True

    def train(self, epochs=None, round_num=None):
        """Train the model on client data.

        Args:
            epochs: Number of epochs to train (defaults to self.epochs)
            round_num: The current round number

        Returns:
            dict: Dictionary containing training results
        """
        epochs = epochs or self.epochs

        print(f"\n=== CLIENT {self.client_id} TRAINING FOR ROUND {round_num} ===")

        # Train primary model
        print("Training primary model...")
        training_results = train_model(self, epochs)

        # Add round number to the training results
        if round_num is not None:
            training_results["round"] = round_num

        save_training_results(self, training_results, round_num)

        # Fix the formatting error by checking if accuracy is a number before formatting
        accuracy = training_results.get('accuracy', 'N/A')
        accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (float, int)) else accuracy
        print(f"Primary model training complete. Accuracy: {accuracy_str}")

        # Train background models if any
        if hasattr(self, 'background_tracks') and self.background_tracks:
            print(f"Client {self.client_id} has {len(self.background_tracks)} background tracks to train")

            for bg_track in self.background_tracks:
                print(f"Training on background track: '{bg_track['name']}'")

                # Save the current primary model state
                primary_state = self.model.state_dict()

                # Set the model to the background track model
                self.model = bg_track['model']

                # Train on this background track
                bg_results = train_model(self, epochs)

                # Don't save background training results to avoid confusion
                # But save the trained background model when saving models later

                # This model will be saved when save_round_model is called
                bg_track['trained'] = True

                # Fix the formatting error by checking if accuracy is a number before formatting
                bg_accuracy = bg_results.get('accuracy', 'N/A')
                bg_accuracy_str = f"{bg_accuracy:.4f}" if isinstance(bg_accuracy, (float, int)) else bg_accuracy
                print(f"Background model '{bg_track['name']}' training complete. Accuracy: {bg_accuracy_str}")

                # Restore primary model
                self.model.load_state_dict(primary_state)
        else:
            print(f"Client {self.client_id} has no background tracks to train")

        print(f"=== END CLIENT {self.client_id} TRAINING ===\n")
        return training_results

    def save_round_model(self, round_num):
        """Save the trained model for a specific round.

        Args:
            round_num: The current round number

        Returns:
            str: Path to the saved model directory
        """
        if not self.results_dir:
            return None

        print(f"\n=== CLIENT {self.client_id} MODEL SAVING FOR ROUND {round_num} ===")

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Create the client directory for this round
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        clients_dir = os.path.join(round_dir, structure["clients_dir"])
        os.makedirs(clients_dir, exist_ok=True)

        client_dir = os.path.join(clients_dir, f"{structure['client_prefix']}{self.client_id}")
        os.makedirs(client_dir, exist_ok=True)

        # Save the primary model
        self.save_model(client_dir)
        print(f"Saved primary model to {client_dir}")

        # Save background models if any were trained
        bg_models_saved = 0
        if hasattr(self, 'background_tracks') and self.background_tracks:
            for bg_track in self.background_tracks:
                if bg_track.get('trained', False):
                    print(f"Saving trained background model for track: '{bg_track['name']}'")

                    # Create a special directory for this background model
                    bg_dir = os.path.join(client_dir, f"background_{bg_track['name']}")
                    os.makedirs(bg_dir, exist_ok=True)

                    # Save model state dict
                    model_path = os.path.join(bg_dir, "model.pt")
                    torch.save(bg_track['model'].state_dict(), model_path)

                    # Save model metadata
                    metadata = {
                        "client_id": self.client_id,
                        "experiment_type": self.experiment_type,
                        "track_name": bg_track['name'],
                        "is_background": True,
                        "timestamp": datetime.now().isoformat()
                    }

                    metadata_path = os.path.join(bg_dir, "metadata.json")
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                    bg_models_saved += 1

        print(f"Saved {bg_models_saved} background models")
        print(f"=== END CLIENT {self.client_id} MODEL SAVING ===\n")
        return client_dir

    def get_model_parameters(self):
        """Get model parameters to send to the server.

        Returns:
            list: List of model parameter tensors
        """
        return self.model.get_parameters()

    def set_model_parameters(self, parameters):
        """Update model with parameters from the server.

        Args:
            parameters: List of model parameter tensors
        """
        self.model.set_parameters(parameters)
        print(f"Client {self.client_id} updated model with parameters from server")

    def _get_structure_config(self):
        """Get the directory structure configuration.

        Returns:
            dict: Directory structure configuration
        """
        # Default structure configuration
        default_structure = {
            "model_storage_dir": "model_storage",
            "round_template": "model_storage/round_{round}",
            "clients_dir": "clients",
            "global_model": "global_model_for_training",
            "client_prefix": "client_"
        }

        # Try to load from configuration file
        config_path = os.path.join(os.path.dirname(self.results_dir), "mock_etcd/configuration.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "results" in config and "structure" in config["results"]:
                        return config["results"]["structure"]
        except Exception as e:
            print(f"Error loading configuration: {e}")

        return default_structure
