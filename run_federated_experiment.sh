#!/bin/bash

# Script to run federated learning experiments

# Display help
usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -e, --experiment <name>  Experiment type (n_cmapss or mnist)"
  echo "  -c, --clients <ids>      Client IDs (e.g., '0 1 2 3 4 5')"
  echo "  -r, --rounds <num>       Number of FL rounds (default: 3)"
  echo "  -l, --local-epochs <num> Number of local epochs (default: 5)"
  echo "  -b, --batch-size <num>   Batch size (default: 64)"
  echo "  -s, --setup-data         Setup data (for MNIST only)"
  echo "  -i, --iid                Use IID data distribution (for MNIST only)"
  echo "  -h, --help               Display this help and exit"
  echo
  echo "Examples:"
  echo "  $0 -e n_cmapss -c '0 1 2' -r 5"
  echo "  $0 -e mnist -c '0 1 2 3 4 5' -s -i"
  exit 1
}

# Default values
EXPERIMENT="n_cmapss"
CLIENTS="0 1 2 3 4 5"
ROUNDS=3
LOCAL_EPOCHS=5
BATCH_SIZE=64
SETUP_DATA=""
IID=""

# Parse arguments
while [ "$1" != "" ]; do
  case $1 in
    -e | --experiment )
      shift
      EXPERIMENT=$1
      ;;
    -c | --clients )
      shift
      CLIENTS=$1
      ;;
    -r | --rounds )
      shift
      ROUNDS=$1
      ;;
    -l | --local-epochs )
      shift
      LOCAL_EPOCHS=$1
      ;;
    -b | --batch-size )
      shift
      BATCH_SIZE=$1
      ;;
    -s | --setup-data )
      SETUP_DATA="--setup_data"
      ;;
    -i | --iid )
      IID="--iid"
      ;;
    -h | --help )
      usage
      exit
      ;;
    * )
      usage
      exit 1
  esac
  shift
done

# Validate experiment type
if [ "$EXPERIMENT" != "n_cmapss" ] && [ "$EXPERIMENT" != "mnist" ]; then
  echo "Error: Experiment type must be 'n_cmapss' or 'mnist'"
  usage
  exit 1
fi

# Setup data for MNIST if requested
if [ "$EXPERIMENT" = "mnist" ] && [ "$SETUP_DATA" = "--setup_data" ]; then
  echo "Setting up MNIST data..."

  if [ "$IID" = "--iid" ]; then
    uv run setup_mnist_data.py --iid
  else
    uv run setup_mnist_data.py
  fi

  echo "MNIST data setup complete."
fi

# Run the federated learning experiment
echo "Running $EXPERIMENT federated learning experiment with clients: $CLIENTS"
echo "Parameters: rounds=$ROUNDS, local_epochs=$LOCAL_EPOCHS, batch_size=$BATCH_SIZE"

uv run fl_orchestrator.py \
  --experiment $EXPERIMENT \
  --clients $CLIENTS \
  --fl_rounds $ROUNDS \
  --local_epochs $LOCAL_EPOCHS \
  --batch_size $BATCH_SIZE \
  $IID

echo "Experiment completed."
