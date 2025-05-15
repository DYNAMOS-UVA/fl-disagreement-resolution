"""Configuration loader for federated learning."""

import os
import json
from datetime import datetime

class ConfigLoader:
    """Loads and manages configuration for federated learning."""

    def __init__(self, config_path="mock_etcd/configuration.json"):
        """Initialize the configuration loader.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.storage_dir = self._setup_storage_dir()

    def _load_config(self):
        """Load configuration from JSON file.

        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {self.config_path}")

    def _setup_storage_dir(self):
        """Setup and return the storage directory path.

        Returns:
            str: Path to the storage directory
        """
        storage_config = self.config.get("storage", {})
        experiment_type = self.config.get("experiment", {}).get("type", "unknown")

        # If a custom directory is specified, use it
        if storage_config.get("custom_dir"):
            storage_dir = storage_config["custom_dir"]
        # Otherwise, use a timestamped directory in the base directory if specified
        elif storage_config.get("use_timestamped_dir", True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            storage_dir = os.path.join(
                storage_config.get("base_dir", "storage"),
                f"fl_simulation_{timestamp}_{experiment_type}"
            )
        # Otherwise, just use the base directory
        else:
            storage_dir = storage_config.get("base_dir", "storage")

        # Create the directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)

        # Create the output directory
        output_dir = os.path.join(
            storage_dir,
            storage_config.get("structure", {}).get("output_dir", "output")
        )
        os.makedirs(output_dir, exist_ok=True)

        # Create output directories for server
        server_output_dir = os.path.join(output_dir, "server")
        os.makedirs(server_output_dir, exist_ok=True)
        os.makedirs(os.path.join(server_output_dir, "plots"), exist_ok=True)

        return storage_dir

    def get_experiment_config(self):
        """Get experiment configuration.

        Returns:
            dict: Experiment configuration
        """
        return self.config.get("experiment", {})

    def get_data_config(self):
        """Get data configuration.

        Returns:
            dict: Data configuration
        """
        return self.config.get("data", {})

    def get_training_config(self):
        """Get training configuration.

        Returns:
            dict: Training configuration
        """
        return self.config.get("training", {})

    def get_storage_config(self):
        """Get storage configuration.

        Returns:
            dict: Storage configuration
        """
        config = self.config.get("storage", {})
        # Add the computed storage directory
        config["storage_dir"] = self.storage_dir
        return config

    def get_path(self, *path_components):
        """Get a path within the storage directory.

        Args:
            *path_components: Components of the path to join

        Returns:
            str: Full path
        """
        return os.path.join(self.storage_dir, *path_components)

    def get_train_dir(self, experiment_type=None):
        """Get the training data directory for the specified experiment type.

        Args:
            experiment_type: Type of experiment (if None, use the one from config)

        Returns:
            str: Path to the training data directory
        """
        if experiment_type is None:
            experiment_type = self.config.get("experiment", {}).get("type")

        train_dirs = self.config.get("data", {}).get("train_dir", {})
        return train_dirs.get(experiment_type)

    def get_test_dir(self, experiment_type=None):
        """Get the test data directory for the specified experiment type.

        Args:
            experiment_type: Type of experiment (if None, use the one from config)

        Returns:
            str: Path to the test data directory
        """
        if experiment_type is None:
            experiment_type = self.config.get("experiment", {}).get("type")

        test_dirs = self.config.get("data", {}).get("test_dir", {})
        return test_dirs.get(experiment_type)
