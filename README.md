# Federated Learning Framework

A modular, flexible framework for federated learning experiments that supports both N-CMAPSS RUL prediction and MNIST classification (with extension support for other datasets).

## Project Structure

- `fl_orchestrator.py`: Central orchestrator that coordinates clients and server
- `fl_client.py`: Client-side code for local model training
- `fl_server.py`: Server-side code for model aggregation and evaluation
- `models.py`: Model definitions for different experiments
- `data_utils.py`: Data loading and preprocessing utilities

## Quick Start

### Run Federated Learning with N-CMAPSS

```bash
python fl_orchestrator.py --experiment n_cmapss --clients 0 1 2 3 4 5 --fl_rounds 3
```

### Run Federated Learning with MNIST

First, you need to set up the MNIST dataset:

```bash
# Download and distribute MNIST data to clients (IID distribution)
python fl_orchestrator.py --experiment mnist --clients 0 1 2 3 4 5 --setup_data --iid

# Download and distribute MNIST data to clients (Non-IID distribution)
python fl_orchestrator.py --experiment mnist --clients 0 1 2 3 4 5 --setup_data
```

Then run the federated learning process:

```bash
# Run with IID data distribution
python fl_orchestrator.py --experiment mnist --clients 0 1 2 3 4 5 --fl_rounds 3 --iid

# Run with Non-IID data distribution
python fl_orchestrator.py --experiment mnist --clients 0 1 2 3 4 5 --fl_rounds 3
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

# MNIST example
python fl_orchestrator.py \
  --experiment mnist \
  --clients 0 1 2 3 4 5 \
  --client_sample_size 1000 \
  --batch_size 64 \
  --local_epochs 5 \
  --lr 0.001 \
  --fl_rounds 5 \
  --iid
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

## Output Structure

- `output/client_results/`: Individual client training results
- `output/server_results/`: Server aggregation and evaluation results
- `output/orchestrator_results/`: Overall federated learning results
- `output/models/`: Saved models for each round
- `output/plots/`: Performance plots and visualizations (confusion matrices for MNIST, RUL predictions for N-CMAPSS)

## Requirements

- PyTorch
- torchvision (for MNIST)
- NumPy
- Matplotlib
- scikit-learn
- seaborn (for visualization)
