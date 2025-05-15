# Federated Learning with Disagreement Resolution

## Track-Specific Evaluation and Visualization

The federated learning system supports track-specific evaluation, where each separate model track created due to client disagreements is evaluated individually. This provides insights into how the different tracks perform compared to the global model.

Key features:
- Each track's performance metrics (accuracy, precision, recall, F1 score) are evaluated and displayed at the end of each round
- Track comparison plots show the relative performance of each track within a round
- Track progress plots visualize how each track's performance evolves over time
- Detailed track evaluation results are saved as JSON files for further analysis

The evaluation system automatically detects when multiple tracks are created due to client disagreements and evaluates each track's model independently against the test dataset.

# Federated Learning Framework

A modular, flexible framework for federated learning experiments that supports both N-CMAPSS RUL prediction and MNIST classification (with extension support for other datasets).

## Project Structure

- `fl_orchestrator.py`: Central orchestrator that coordinates clients and server
- `fl_client/`: Client-side implementation for local model training
- `fl_server/`: Server-side implementation for model aggregation and evaluation
- `fl_module/`: Models, data loading and preprocessing utilities
  - `fl_module/n_cmapss/`: N-CMAPSS dataset utilities
  - `fl_module/mnist/`: MNIST dataset utilities
- `mock_etcd/`: Configuration management
- `run_federated_experiment.sh`: Convenient shell script to run experiments
- `results/`: Directory for storing models and outputs during federated learning simulation

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

# Run with a custom results directory
./run_federated_experiment.sh -e mnist -c "0 1 2" -r 2 -d "results/my_experiment"

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
# This will be handled automatically when you run the orchestrator
# with --setup_data flag
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
  --fl_rounds 5 \
  --local_epochs 5

# MNIST example with custom results directory
python fl_orchestrator.py \
  --experiment mnist \
  --clients 0 1 2 3 4 5 \
  --fl_rounds 5 \
  --local_epochs 5 \
  --iid \
  --results_dir results/my_experiment
```

## Component Usage

### Running Individual Clients

You can run individual clients separately:

```bash
# N-CMAPSS client
python -m fl_client.main --client_id 0 --experiment n_cmapss

# MNIST client
python -m fl_client.main --client_id 0 --experiment mnist
```

### Running the Server

You can run the server alone for testing:

```bash
# N-CMAPSS server
python -m fl_server.main --experiment n_cmapss --test_units 11 14 15

# MNIST server
python -m fl_server.main --experiment mnist
```

## Data Distribution in MNIST

The framework supports two types of data distribution for MNIST:

- **IID (Independent and Identically Distributed)**: Each client gets a random subset of the MNIST dataset with similar class distributions.
- **Non-IID**: Each client gets a biased subset of data with different class distributions:
  - Each client has 2 primary classes (70% of data)
  - Each client has 3 secondary classes (30% of data)
  - This simulates real-world scenarios where clients have different data distributions

## Results Structure

By default, each experiment creates a timestamped directory under `
