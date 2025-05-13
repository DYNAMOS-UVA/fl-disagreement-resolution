import torch
import torch.nn as nn

# Base model class
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def get_parameters(self):
        return [param.data.clone() for param in self.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.parameters(), parameters):
            param.data = new_param.clone()

# N-CMAPSS RUL prediction model
class RULPredictor(BaseModel):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(RULPredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Simple CNN model for MNIST (for future use)
class MNISTClassifier(BaseModel):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Factory function to create model based on experiment type
def create_model(experiment_type, **kwargs):
    if experiment_type == "n_cmapss":
        return RULPredictor(**kwargs)
    elif experiment_type == "mnist":
        return MNISTClassifier()
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
