"""MNIST dataset module for image classification tasks."""

from data_module.mnist.dataset import MNISTDataset
from data_module.mnist.utils import (
    setup_federated_data,
    load_client_data,
    load_test_data,
    create_client_dataloaders,
    create_test_dataloader
)
