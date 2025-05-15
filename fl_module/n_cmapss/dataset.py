"""N-CMAPSS dataset class for remaining useful life (RUL) prediction."""

import torch
from fl_module.base import BaseDataset

class NCMAPSSDataset(BaseDataset):
    """Dataset class for N-CMAPSS RUL prediction."""

    def __init__(self, samples, labels):
        """Initialize the dataset with samples and labels.

        Args:
            samples: Numpy array of shape (n_samples, seq_len, n_features)
            labels: Numpy array of shape (n_samples,)
        """
        super(NCMAPSSDataset, self).__init__()
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a sample from the dataset.

        Args:
            idx: Index of the sample to return

        Returns:
            tuple: (sample, label) where sample has shape (seq_len, n_features)
                  and label has shape (1,)
        """
        return self.samples[idx], self.labels[idx]
