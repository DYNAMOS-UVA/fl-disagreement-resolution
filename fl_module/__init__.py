"""
Federated Learning module for data, models, and utilities.

This module contains dataset-specific utilities for loading and preprocessing
data for different experiments, as well as model definitions and other utilities
for federated learning.
"""

# Import main classes and functions from submodules
from fl_module.base import BaseDataset
from fl_module.n_cmapss.dataset import NCMAPSSDataset
from fl_module.mnist.dataset import MNISTDataset

# Import model classes and functions
from fl_module.models import (
    BaseModel,
    RULPredictor,
    MNISTClassifier,
    create_model
)

# Import utility functions
from fl_module.n_cmapss.utils import (
    load_client_data as load_ncmapss_client_data,
    load_test_data as load_ncmapss_test_data,
    preprocess_data as preprocess_ncmapss_data,
    create_client_dataloaders as create_ncmapss_client_dataloaders,
    create_test_dataloader as create_ncmapss_test_dataloader
)

from fl_module.mnist.utils import (
    setup_federated_data as setup_mnist_federated_data,
    load_client_data as load_mnist_client_data,
    load_test_data as load_mnist_test_data,
    create_client_dataloaders as create_mnist_client_dataloaders,
    create_test_dataloader as create_mnist_test_dataloader
)
