"""Base dataset classes and utilities."""

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base dataset class for all experiments.

    All dataset classes should inherit from this class and implement
    the __len__ and __getitem__ methods.
    """

    def __init__(self):
        """Initialize the dataset."""
        super(BaseDataset, self).__init__()

    def __len__(self):
        """Return the number of samples in the dataset."""
        raise NotImplementedError("Dataset class must implement __len__")

    def __getitem__(self, idx):
        """Return a sample from the dataset at the given index."""
        raise NotImplementedError("Dataset class must implement __getitem__")
