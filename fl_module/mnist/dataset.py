"""MNIST dataset class for image classification."""

import torch
from fl_module.base import BaseDataset

class MNISTDataset(BaseDataset):
    """Dataset class for MNIST image classification."""

    def __init__(self, images, labels):
        """Initialize the dataset with images and labels.

        Args:
            images: Numpy array of shape (n_samples, channels, height, width)
            labels: Numpy array of shape (n_samples,)
        """
        super(MNISTDataset, self).__init__()
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a sample from the dataset.

        Args:
            idx: Index of the sample to return

        Returns:
            tuple: (image, label) where image has shape (channels, height, width)
                  and label is a scalar
        """
        return self.images[idx], self.labels[idx]
