"""Utility functions for MNIST dataset."""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

from data_module.mnist.dataset import MNISTDataset

def download_mnist_dataset(data_dir='data/mnist'):
    """Download MNIST dataset if not already available.

    Args:
        data_dir: Directory to store the dataset

    Returns:
        tuple: (train_dataset, test_dataset) from torchvision.datasets.MNIST
    """
    os.makedirs(data_dir, exist_ok=True)

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    print(f"MNIST dataset downloaded to {data_dir}")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset

def distribute_mnist_to_clients(train_dataset, num_clients=6, samples_per_client=1000, iid=False, data_dir='data/mnist'):
    """Distribute MNIST data to multiple clients.

    Args:
        train_dataset: MNIST training dataset from torchvision
        num_clients: Number of clients to distribute data to
        samples_per_client: Maximum number of samples per client
        iid: Whether to use IID (Independent and Identically Distributed) data distribution
        data_dir: Directory to store client data
    """
    # Create client directories
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)

    # Get all data
    all_data = []
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        all_data.append((img, label))

    # Distribute data to clients
    client_data = [[] for _ in range(num_clients)]

    if iid:
        # Shuffle data randomly for IID distribution
        random.shuffle(all_data)
        # Distribute evenly
        for i, item in enumerate(all_data):
            if i < num_clients * samples_per_client:
                client_data[i % num_clients].append(item)
    else:
        # Non-IID: sort by label, then allocate different distributions to each client
        sorted_data = [[] for _ in range(10)]  # 10 classes in MNIST
        for img, label in all_data:
            sorted_data[label].append((img, label))

        # Different primary class distributions for each client
        primary_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 9]][:num_clients]
        secondary_classes = [[2, 3, 4], [0, 4, 6], [1, 2, 8], [0, 3, 9], [1, 5, 7], [3, 6, 8]][:num_clients]

        for client_id in range(num_clients):
            # Add more samples from primary classes (70%)
            primary_samples = int(samples_per_client * 0.7)
            for cls in primary_classes[client_id]:
                samples_from_class = min(primary_samples // len(primary_classes[client_id]), len(sorted_data[cls]))
                client_data[client_id].extend(sorted_data[cls][:samples_from_class])
                sorted_data[cls] = sorted_data[cls][samples_from_class:]

            # Add some samples from secondary classes (30%)
            secondary_samples = samples_per_client - len(client_data[client_id])
            for cls in secondary_classes[client_id]:
                if secondary_samples <= 0:
                    break
                samples_from_class = min(secondary_samples // len(secondary_classes[client_id]), len(sorted_data[cls]))
                client_data[client_id].extend(sorted_data[cls][:samples_from_class])
                sorted_data[cls] = sorted_data[cls][samples_from_class:]
                secondary_samples -= samples_from_class

    # Save data for each client
    for client_id in range(num_clients):
        client_dir = os.path.join(data_dir, 'train', f'client_{client_id}')
        os.makedirs(client_dir, exist_ok=True)

        images = []
        labels = []
        for img, label in client_data[client_id]:
            # Convert from tensor to numpy
            images.append(img.numpy())
            labels.append(label)

        # Save as .npz file
        data_path = os.path.join(client_dir, 'mnist_data.npz')
        np.savez(
            data_path,
            images=np.array(images),
            labels=np.array(labels)
        )

        # Count classes
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Client {client_id} has {len(labels)} samples. Class distribution: {class_dist}")

    print(f"MNIST data distributed to {num_clients} clients")

def prepare_mnist_test_data(test_dataset, data_dir='data/mnist'):
    """Prepare MNIST test data.

    Args:
        test_dataset: MNIST test dataset from torchvision
        data_dir: Directory to store test data
    """
    os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)

    # Get all test data
    images = []
    labels = []
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        images.append(img.numpy())
        labels.append(label)

    # Save test data
    test_data_path = os.path.join(data_dir, 'test', 'mnist_test.npz')
    np.savez(
        test_data_path,
        images=np.array(images),
        labels=np.array(labels)
    )

    print(f"MNIST test data prepared with {len(labels)} samples")

def setup_federated_data(num_clients=6, samples_per_client=1000, iid=False, data_dir='data/mnist'):
    """Download and setup MNIST data for federated learning.

    Args:
        num_clients: Number of clients to distribute data to
        samples_per_client: Maximum number of samples per client
        iid: Whether to use IID data distribution
        data_dir: Directory to store the dataset
    """
    # Check if data is already distributed to clients
    train_client_data_exists = all(os.path.exists(os.path.join(data_dir, 'train', f'client_{i}', 'mnist_data.npz'))
                                 for i in range(num_clients))
    test_data_exists = os.path.exists(os.path.join(data_dir, 'test', 'mnist_test.npz'))

    if train_client_data_exists and test_data_exists:
        print(f"MNIST data already exists for {num_clients} clients.")
        client_samples = []
        for i in range(num_clients):
            data_path = os.path.join(data_dir, 'train', f'client_{i}', 'mnist_data.npz')
            data = np.load(data_path)
            labels = data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            class_dist = dict(zip(unique, counts))
            print(f"Client {i} has {len(labels)} samples. Class distribution: {class_dist}")
            client_samples.append(len(labels))

        print(f"Test data exists at {os.path.join(data_dir, 'test', 'mnist_test.npz')}")
        print(f"Using existing MNIST data with distribution type: {'IID' if iid else 'Non-IID'}")
        print(f"Clients have an average of {sum(client_samples) / len(client_samples):.0f} samples each")
        return

    # Download MNIST dataset
    train_dataset, test_dataset = download_mnist_dataset(data_dir)

    # Distribute to clients
    distribute_mnist_to_clients(
        train_dataset,
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        iid=iid,
        data_dir=data_dir
    )

    # Prepare test data
    prepare_mnist_test_data(test_dataset, data_dir)

    print(f"MNIST data setup completed for federated learning with {num_clients} clients")

def load_client_data(client_id, train_dir='data/mnist/train', sample_size=None):
    """Load MNIST data for a specific client.

    Args:
        client_id: Client ID (0-5)
        train_dir: Directory containing training data
        sample_size: Maximum number of samples to load (optional)

    Returns:
        tuple: (images, labels) where images has shape (n_samples, channels, height, width)
               and labels has shape (n_samples,)
    """
    data_path = os.path.join(train_dir, f'client_{client_id}', 'mnist_data.npz')
    print(f"Loading MNIST data from {data_path}")

    data = np.load(data_path)
    images = data['images']
    labels = data['labels']

    # Sample if needed
    if sample_size and len(images) > sample_size:
        print(f"Sampling {sample_size} instances from {len(images)} for client {client_id}")
        indices = np.random.choice(len(images), sample_size, replace=False)
        images = images[indices]
        labels = labels[indices]

    print(f"Client {client_id} loaded {len(images)} MNIST samples")
    return images, labels

def load_test_data(test_dir='data/mnist/test'):
    """Load MNIST test data.

    Args:
        test_dir: Directory containing test data

    Returns:
        tuple: (images, labels) where images has shape (n_samples, channels, height, width)
               and labels has shape (n_samples,)
    """
    data_path = os.path.join(test_dir, 'mnist_test.npz')
    print(f"Loading MNIST test data from {data_path}")

    data = np.load(data_path)
    images = data['images']
    labels = data['labels']

    print(f"Loaded {len(images)} MNIST test samples")
    return images, labels

def create_client_dataloaders(images, labels, batch_size=64, valid_split=0.2):
    """Create dataloaders for MNIST training and validation.

    Args:
        images: Training images of shape (n_samples, channels, height, width)
        labels: Training labels of shape (n_samples,)
        batch_size: Batch size for training
        valid_split: Fraction of data to use for validation

    Returns:
        tuple: (train_loader, valid_loader)
    """
    # Split into train and validation
    split_idx = int(len(images) * (1 - valid_split))

    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    valid_images = images[split_idx:]
    valid_labels = labels[split_idx:]

    # Create datasets
    train_dataset = MNISTDataset(train_images, train_labels)
    valid_dataset = MNISTDataset(valid_images, valid_labels)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_loader, valid_loader

def create_test_dataloader(images, labels, batch_size=64):
    """Create dataloader for MNIST test data.

    Args:
        images: Test images of shape (n_samples, channels, height, width)
        labels: Test labels of shape (n_samples,)
        batch_size: Batch size for testing

    Returns:
        DataLoader: Test dataloader
    """
    test_dataset = MNISTDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader
