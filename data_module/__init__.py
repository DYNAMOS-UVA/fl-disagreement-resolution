"""
Data loading and preprocessing module for the federated learning framework.

This module contains dataset-specific utilities for loading and preprocessing
data for different experiments.
"""

# Import main classes and functions from submodules
from data_module.base import BaseDataset
from data_module.n_cmapss.dataset import NCMAPSSDataset
from data_module.mnist.dataset import MNISTDataset

# Import utility functions
from data_module.n_cmapss.utils import (
    load_client_data as load_ncmapss_client_data,
    load_test_data as load_ncmapss_test_data,
    preprocess_data as preprocess_ncmapss_data,
    create_client_dataloaders as create_ncmapss_client_dataloaders,
    create_test_dataloader as create_ncmapss_test_dataloader
)

from data_module.mnist.utils import (
    setup_federated_data as setup_mnist_federated_data,
    load_client_data as load_mnist_client_data,
    load_test_data as load_mnist_test_data,
    create_client_dataloaders as create_mnist_client_dataloaders,
    create_test_dataloader as create_mnist_test_dataloader
)
