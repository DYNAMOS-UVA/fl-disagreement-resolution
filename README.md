# Federated Learning Framework

A modular, flexible framework for federated learning experiments that supports both N-CMAPSS RUL prediction and MNIST classification (with extension support for other datasets).

## Project Structure

- `fl_orchestrator.py`: Central orchestrator that coordinates clients and server
- `fl_client.py`: Client-side code for local model training
- `fl_server.py`: Server-side code for model aggregation and evaluation
- `models.py`: Model definitions for different experiments
- `data_module/`: Modular data loading and preprocessing utilities
  - `data_module/n_cmapss/`: N-CMAPSS dataset utilities
  - `data_module/mnist/`: MNIST dataset utilities
- `setup_mnist_data.py`: Utility script to set up MNIST data for federated learning
- `run_federated_experiment.sh`: Convenient shell script to run experiments
- `storage/`: Directory for storing models and results during federated learning simulation

## Features

- Support for multiple datasets (N-CMAPSS and MNIST)
- Modular architecture for easy extension
- File-based model communication (simulating distributed environments)
- IID and non-IID data distribution for MNIST
- Smart data management (download once, reuse for multiple experiments)
- Comprehensive metrics and visualizations:
  - N-CMAPSS: RMSE, MAE, R², % within tolerance ranges, color-coded plots
  - MNIST: Accuracy, precision, recall, F1 score, per-class performance, enhanced visualizations

## Metrics and Visualizations

### N-CMAPSS (RUL Prediction)

The framework provides rich metrics for the N-CMAPSS Remaining Useful Life (RUL) prediction task:

- **RMSE (Root Mean Squared Error)**: Traditional error metric (lower is better)
- **MAE (Mean Absolute Error)**: Average absolute difference in cycles (lower is better)
- **R² (Coefficient of Determination)**: How well the model explains the variance (higher is better, max 1.0)
- **Within ±10 cycles**: Percentage of predictions within 10 cycles of actual RUL (higher is better)
- **Within ±20 cycles**: Percentage of predictions within 20 cycles of actual RUL (higher is better)

Visualizations include:
- Color-coded scatter plots (green: within 10 cycles, orange: within 20 cycles, red: beyond 20 cycles)
- Performance metrics over rounds (RMSE, MAE, R², within tolerance percentages)
- Reference lines showing perfect prediction and tolerance ranges

### MNIST (Image Classification)

For the MNIST classification task, the framework provides:

- **Overall Accuracy**: Percentage of correctly classified images
- **Precision**: Overall precision (weighted) and per-class precision values
- **Recall**: Overall recall (weighted) and per-class recall values
- **F1 Score**: Overall F1 score (weighted) and per-class F1 values
- **Per-class Accuracy**: Accuracy for each individual digit class
- **Confusion Matrix**: Raw counts and normalized matrices to show classification performance across classes
- **Per-round progression**: Tracks how all metrics improve over federated learning rounds

Visualizations include:
- Dual confusion matrices (raw counts and normalized by true label)
- Comprehensive metrics dashboard with accuracy, precision, recall, and F1 score trends
- Per-class metrics bar charts comparing accuracy, precision, recall, and F1 for each digit
- Summary statistics displayed within visualizations for quick interpretation

## Quick Start

### Using the Convenience Script

The easiest way to run experiments is with the provided shell script:

```bash
# Run N-CMAPSS experiment
./run_federated_experiment.sh -e n_cmapss -c "0 1 2 3 4 5" -r 3

# Run MNIST experiment with data setup and IID distribution
./run_federated_experiment.sh -e mnist -c "0 1 2 3 4 5" -r 3 -s -i

# Run with a custom storage directory
./run_federated_experiment.sh -e mnist -c "0 1 2" -r 2 -d "storage/my_experiment"

# Force recreate MNIST data with IID distribution
./run_federated_experiment.sh -e mnist -c "0 1 2 3" -s -f -i

# Get help for script options
./run_federated_experiment.sh --help
```

### Run Federated Learning with N-CMAPSS

```bash
python fl_orchestrator.py --experiment n_cmapss --clients 0 1 2 3 4 5 --fl_rounds 3
```

### Run Federated Learning with MNIST

First, you need to set up the MNIST dataset (if you haven't already):

```bash
# Download and distribute MNIST data to clients (IID distribution)
python setup_mnist_data.py --iid

# Download and distribute MNIST data to clients (Non-IID distribution)
python setup_mnist_data.py

# Force recreation of MNIST data
python setup_mnist_data.py --force
```

Then run the federated learning process:

```bash
# Run with IID data distribution
python fl_orchestrator.py --experiment mnist --clients 0 1 2 3 4 5 --fl_rounds 3 --iid

# Run with Non-IID data distribution
python fl_orchestrator.py --experiment mnist --clients 0 1 2 3 4 5 --fl_rounds 3

# Run with data setup (will only download if data doesn't exist)
python fl_orchestrator.py --experiment mnist --clients 0 1 2 3 4 5 --fl_rounds 3 --setup_data

# Force recreate data even if it exists
python fl_orchestrator.py --experiment mnist --clients 0 1 2 3 4 5 --fl_rounds 3 --setup_data --force_setup_data
```

### Customize Your Run

```bash
# N-CMAPSS example
python fl_orchestrator.py \
  --experiment n_cmapss \
  --clients 0 1 2 \
  --train_dir data/n-cmapss/train \
  --test_dir data/n-cmapss/test \
  --test_units 11 14 15 \
  --client_sample_size 1000 \
  --test_sample_size 500 \
  --batch_size 64 \
  --local_epochs 5 \
  --lr 0.001 \
  --fl_rounds 5

# MNIST example with custom storage directory
python fl_orchestrator.py \
  --experiment mnist \
  --clients 0 1 2 3 4 5 \
  --client_sample_size 1000 \
  --batch_size 64 \
  --local_epochs 5 \
  --lr 0.001 \
  --fl_rounds 5 \
  --iid \
  --storage_dir storage/my_experiment
```

## Component Usage

### Running Individual Clients

You can run individual clients separately:

```bash
# N-CMAPSS client
python fl_client.py --client_id 0 --experiment n_cmapss

# MNIST client
python fl_client.py --client_id 0 --experiment mnist
```

### Running the Server

You can run the server alone for testing:

```bash
# N-CMAPSS server
python fl_server.py --experiment n_cmapss --test_units 11 14 15

# MNIST server
python fl_server.py --experiment mnist
```

## Data Distribution in MNIST

The framework supports two types of data distribution for MNIST:

- **IID (Independent and Identically Distributed)**: Each client gets a random subset of the MNIST dataset with similar class distributions.
- **Non-IID**: Each client gets a biased subset of data with different class distributions:
  - Each client has 2 primary classes (70% of data)
  - Each client has 3 secondary classes (30% of data)
  - This simulates real-world scenarios where clients have different data distributions

## Storage Structure

By default, each experiment creates a timestamped directory under `storage/` with the following structure:

```
storage/fl_simulation_YYYYMMDD_HHMMSS/
├── global_model_initial/                # Initial global model
├── round_1/                             # Data for round 1
│   ├── clients/                         # Client models
│   │   ├── client_0/                    # Model from client 0
│   │   │   ├── metadata.json            # Model metadata
│   │   │   └── model.pt                 # Trained model weights
│   │   ├── client_1/
│   │   └── ...
│   ├── global_model/                    # Global model distributed to clients
│   └── global_model_aggregated/         # Aggregated model after client training
├── round_2/
│   └── ...
└── output/                              # Experiment output
    ├── clients/                         # Client-specific results
    │   └── ...
    ├── fl_results.json                  # Overall experiment results
    ├── models/                          # Saved models for each round
    │   └── ...
    └── server/                          # Server results
        ├── plots/                       # Performance visualizations
        │   └── ...
        └── training_history_round_*.json # Server training history
```

You can specify a custom storage directory with the `--storage-dir` option.

## Output Structure

- `output/client_results/`: Individual client training results (when not using storage dir)
- `output/server_results/`: Server aggregation and evaluation results (when not using storage dir)
- `output/orchestrator_results/`: Overall federated learning results (when not using storage dir)
- `output/models/`: Saved models for each round (when not using storage dir)
- `output/plots/`: Performance plots and visualizations (when not using storage dir)

## Extending the Framework

To add a new dataset or experiment type:

1. Create a new module in `data_module/` for your dataset
2. Implement the dataset class, and data loading/preprocessing utilities
3. Add a new model class in `models.py`
4. Update the orchestrator, client, and server to support the new experiment type

## Requirements

- PyTorch
- torchvision (for MNIST)
- NumPy
- Matplotlib
- scikit-learn
- seaborn (for visualization)
