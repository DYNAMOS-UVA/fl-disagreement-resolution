"""Utility functions for N-CMAPSS dataset."""

import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from fl_module.n_cmapss.dataset import NCMAPSSDataset

# Unit to client mapping
UNIT_TO_CLIENT = {
    2: 0,   # Unit 2 is in client 0
    5: 1,   # Unit 5 is in client 1
    10: 2,  # Unit 10 is in client 2
    16: 3,  # Unit 16 is in client 3
    18: 4,  # Unit 18 is in client 4
    20: 5   # Unit 20 is in client 5
}

def load_client_data(client_id, train_dir, sample_size=1000):
    """Load N-CMAPSS data for a specific client.

    Args:
        client_id: Client ID (0-5)
        train_dir: Directory containing training data
        sample_size: Maximum number of samples to load per client

    Returns:
        tuple: (samples, labels) where samples has shape (n_samples, seq_len, n_features)
               and labels has shape (n_samples,)
    """
    # Find unit for this client
    unit = None
    for u, c in UNIT_TO_CLIENT.items():
        if c == client_id:
            unit = u
            break

    if unit is None:
        raise ValueError(f"No unit found for client {client_id}")

    # Load data
    npz_file = f"Unit{unit}_win50_str1_smp10.npz"
    data_path = os.path.join(train_dir, f"client_{client_id}", npz_file)
    print(f"Loading data from {data_path}")

    data = np.load(data_path)

    # Extract samples and labels, reshape from (window_size, n_features, n_samples) to (n_samples, window_size, n_features)
    samples = data['sample'].transpose(2, 0, 1)
    labels = data['label']

    # Sample to reduce memory usage if needed
    if len(samples) > sample_size:
        print(f"Sampling {sample_size} instances from {len(samples)} for client {client_id}")
        indices = np.random.choice(len(samples), sample_size, replace=False)
        samples = samples[indices]
        labels = labels[indices]
    else:
        print(f"Using all {len(samples)} instances for client {client_id}")

    return samples, labels

def load_test_data(test_dir, test_units, sample_size=500):
    """Load N-CMAPSS test data.

    Args:
        test_dir: Directory containing test data
        test_units: List of unit IDs to use for testing
        sample_size: Maximum number of samples to load per test unit

    Returns:
        tuple: (samples, labels) where samples has shape (n_samples, seq_len, n_features)
               and labels has shape (n_samples,)
    """
    test_samples = []
    test_labels = []

    for unit in test_units:
        npz_file = f"Unit{unit}_win50_str1_smp10.npz"
        data_path = os.path.join(test_dir, npz_file)
        print(f"Loading test data from {data_path}")

        data = np.load(data_path)

        # Extract and reshape samples and labels
        unit_samples = data['sample'].transpose(2, 0, 1)
        unit_labels = data['label']

        # Sample if needed
        if len(unit_samples) > sample_size:
            print(f"Sampling {sample_size} instances from {len(unit_samples)} for test unit {unit}")
            indices = np.random.choice(len(unit_samples), sample_size, replace=False)
            unit_samples = unit_samples[indices]
            unit_labels = unit_labels[indices]
        else:
            print(f"Using all {len(unit_samples)} instances for test unit {unit}")

        test_samples.append(unit_samples)
        test_labels.append(unit_labels)

    # Concatenate all test data
    test_samples = np.vstack(test_samples)
    test_labels = np.concatenate(test_labels)

    return test_samples, test_labels

def preprocess_data(train_samples, test_samples=None):
    """Normalize N-CMAPSS data using StandardScaler.

    Args:
        train_samples: Training samples of shape (n_samples, seq_len, n_features)
        test_samples: Optional test samples of shape (n_samples, seq_len, n_features)

    Returns:
        If test_samples is None:
            tuple: (normalized_train_samples, scaler)
        If test_samples is not None:
            tuple: (normalized_train_samples, normalized_test_samples, scaler)
    """
    # Get shapes
    n_train_samples, seq_len, n_features = train_samples.shape

    # Reshape for normalization
    train_flat = train_samples.reshape(-1, n_features)

    # Create and fit scaler
    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_flat)

    # Reshape back
    train_normalized = train_flat.reshape(n_train_samples, seq_len, n_features)

    # Also normalize test data if provided
    if test_samples is not None:
        n_test_samples = test_samples.shape[0]
        test_flat = test_samples.reshape(-1, n_features)
        test_flat = scaler.transform(test_flat)
        test_normalized = test_flat.reshape(n_test_samples, seq_len, n_features)
        return train_normalized, test_normalized, scaler

    return train_normalized, scaler

def create_client_dataloaders(train_samples, train_labels, batch_size=64, valid_split=0.2):
    """Create dataloaders for client training.

    Args:
        train_samples: Training samples of shape (n_samples, seq_len, n_features)
        train_labels: Training labels of shape (n_samples,)
        batch_size: Batch size for training
        valid_split: Fraction of data to use for validation

    Returns:
        tuple: (train_loader, valid_loader)
    """
    # Split into train and validation
    split_idx = int(len(train_samples) * (1 - valid_split))

    train_data = train_samples[:split_idx]
    train_labels_split = train_labels[:split_idx]
    valid_data = train_samples[split_idx:]
    valid_labels = train_labels[split_idx:]

    # Create datasets
    train_dataset = NCMAPSSDataset(train_data, train_labels_split)
    valid_dataset = NCMAPSSDataset(valid_data, valid_labels)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_loader, valid_loader

def create_test_dataloader(test_samples, test_labels, batch_size=64):
    """Create dataloader for test data.

    Args:
        test_samples: Test samples of shape (n_samples, seq_len, n_features)
        test_labels: Test labels of shape (n_samples,)
        batch_size: Batch size for testing

    Returns:
        DataLoader: Test dataloader
    """
    test_dataset = NCMAPSSDataset(test_samples, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader
