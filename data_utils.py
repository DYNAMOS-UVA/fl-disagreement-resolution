import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torchvision
from torchvision import datasets, transforms
import random

# N-CMAPSS dataset class
class NCMAPSSDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# Load N-CMAPSS data for a specific client
def load_client_data(client_id, train_dir, sample_size=1000):
    # Mapping from unit to client
    unit_to_client = {
        2: 0, 5: 1, 10: 2, 16: 3, 18: 4, 20: 5
    }

    # Find unit for this client
    unit = None
    for u, c in unit_to_client.items():
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

# Load N-CMAPSS test data
def load_test_data(test_dir, test_units, sample_size=500):
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

# Preprocess N-CMAPSS data (normalize)
def preprocess_ncmapss_data(train_samples, test_samples=None):
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

# Create dataloaders for client training
def create_client_dataloaders(train_samples, train_labels, batch_size=64, valid_split=0.2):
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

# Create dataloader for test data
def create_test_dataloader(test_samples, test_labels, batch_size=64):
    test_dataset = NCMAPSSDataset(test_samples, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader

# --------------------------------
# MNIST Dataset Functions
# --------------------------------

def download_mnist_dataset(data_dir='data/mnist'):
    """Download MNIST dataset if not already available"""
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
    """Distribute MNIST data to multiple clients, either IID or non-IID"""
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
        # Shuffle data randomly
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
    """Prepare MNIST test data"""
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

def load_mnist_client_data(client_id, train_dir='data/mnist/train', sample_size=None):
    """Load MNIST data for a specific client"""
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

def load_mnist_test_data(test_dir='data/mnist/test'):
    """Load MNIST test data"""
    data_path = os.path.join(test_dir, 'mnist_test.npz')
    print(f"Loading MNIST test data from {data_path}")

    data = np.load(data_path)
    images = data['images']
    labels = data['labels']

    print(f"Loaded {len(images)} MNIST test samples")
    return images, labels

class MNISTDataset(Dataset):
    """MNIST dataset class"""
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def create_mnist_dataloaders(images, labels, batch_size=64, valid_split=0.2):
    """Create dataloaders for MNIST training and validation"""
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

def create_mnist_test_dataloader(images, labels, batch_size=64):
    """Create dataloader for MNIST test data"""
    test_dataset = MNISTDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader

def setup_mnist_federated_data(num_clients=6, samples_per_client=1000, iid=False, data_dir='data/mnist'):
    """Download and setup MNIST data for federated learning"""
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
