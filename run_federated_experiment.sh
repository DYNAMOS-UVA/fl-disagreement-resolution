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
  echo "  -f, --force-setup        Force data setup even if it exists (for MNIST only)"
  echo "  -i, --iid                Use IID data distribution (for MNIST only)"
  echo "  -d, --results-dir <dir>  Custom results directory (default: auto-generated)"
  echo "  -C, --config <file>      Path to configuration file (default: mock_etcd/configuration.json)"
  echo "  -h, --help               Display this help and exit"
  echo
  echo "Examples:"
  echo "  $0 -e n_cmapss -c '0 1 2' -r 5"
  echo "  $0 -e mnist -c '0 1 2 3 4 5' -s -i"
  echo "  $0 -e mnist -c '0 1 2 3' -d 'results/my_experiment'"
  echo "  $0 -e mnist -c '0 1 2 3' -s -f -i  # Force new data setup with IID distribution"
  echo "  $0 -C custom_config.json           # Use a custom configuration file"
  exit 1
}

# Default values - these will be used to override the config file
EXPERIMENT=""
CLIENTS=""
ROUNDS=""
LOCAL_EPOCHS=""
BATCH_SIZE=""
SETUP_DATA=""
FORCE_SETUP=""
IID=""
RESULTS_DIR=""
CONFIG_FILE="mock_etcd/configuration.json"
OVERRIDE_FLAG=""

# Parse arguments
while [ "$1" != "" ]; do
  case $1 in
    -e | --experiment )
      shift
      EXPERIMENT=$1
      OVERRIDE_FLAG="--override"
      ;;
    -c | --clients )
      shift
      CLIENTS=$1
      OVERRIDE_FLAG="--override"
      ;;
    -r | --rounds )
      shift
      ROUNDS=$1
      OVERRIDE_FLAG="--override"
      ;;
    -l | --local-epochs )
      shift
      LOCAL_EPOCHS=$1
      OVERRIDE_FLAG="--override"
      ;;
    -b | --batch-size )
      shift
      BATCH_SIZE=$1
      OVERRIDE_FLAG="--override"
      ;;
    -s | --setup-data )
      SETUP_DATA="--setup_data"
      OVERRIDE_FLAG="--override"
      ;;
    -f | --force-setup )
      FORCE_SETUP="--force_setup_data"
      OVERRIDE_FLAG="--override"
      ;;
    -i | --iid )
      IID="--iid"
      OVERRIDE_FLAG="--override"
      ;;
    -d | --results-dir )
      shift
      RESULTS_DIR="--results_dir $1"
      OVERRIDE_FLAG="--override"
      ;;
    -C | --config )
      shift
      CONFIG_FILE=$1
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

# Command to run the orchestrator
CMD="uv run fl_orchestrator.py --config $CONFIG_FILE $OVERRIDE_FLAG"

# Add any specified override arguments
if [ "$EXPERIMENT" != "" ]; then
  CMD="$CMD --experiment $EXPERIMENT"
fi

if [ "$CLIENTS" != "" ]; then
  CMD="$CMD --clients $CLIENTS"
fi

if [ "$ROUNDS" != "" ]; then
  CMD="$CMD --fl_rounds $ROUNDS"
fi

if [ "$LOCAL_EPOCHS" != "" ]; then
  CMD="$CMD --local_epochs $LOCAL_EPOCHS"
fi

if [ "$BATCH_SIZE" != "" ]; then
  CMD="$CMD --batch_size $BATCH_SIZE"
fi

if [ "$SETUP_DATA" != "" ]; then
  CMD="$CMD $SETUP_DATA"
fi

if [ "$FORCE_SETUP" != "" ]; then
  CMD="$CMD $FORCE_SETUP"
fi

if [ "$IID" != "" ]; then
  CMD="$CMD $IID"
fi

if [ "$RESULTS_DIR" != "" ]; then
  CMD="$CMD $RESULTS_DIR"
fi

# Print experiment details
if [ "$EXPERIMENT" != "" ]; then
  echo "Running $EXPERIMENT federated learning experiment"
fi

if [ "$CLIENTS" != "" ]; then
  echo "Using clients: $CLIENTS"
fi

echo "Parameters will override configuration in $CONFIG_FILE"
echo "Running command: $CMD"

# Execute the command
eval $CMD

echo "Experiment completed."
