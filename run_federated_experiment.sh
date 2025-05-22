#!/bin/bash

# Script to run federated learning experiments

# Display help
usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -e, --experiment <n>  Experiment type (n_cmapss or mnist)"
  echo "  -c, --clients <ids>      Client IDs (e.g., '0 1 2 3 4 5') or number of clients (e.g., 6)"
  echo "                           If not specified, uses num_clients from scenario file"
  echo "  -r, --rounds <num>       Number of FL rounds (default: 3)"
  echo "  -l, --local-epochs <num> Number of local training epochs (default: 5)"
  echo "  -b, --batch-size <num>   Batch size (default: 64)"
  echo "  -s, --setup-data         Setup data (for MNIST only)"
  echo "  -f, --force-setup        Force data setup even if it exists (for MNIST only)"
  echo "  -i, --iid                Use IID data distribution (for MNIST only)"
  echo "  -d, --results-dir <dir>  Custom results directory (default: auto-generated)"
  echo "  -S, --scenario <num>     Scenario number to run (default: 9 - no disagreements)"
  echo "                           Use 'all' to run all available scenarios sequentially"
  echo "                           Scenarios define num_clients (N-CMAPSS limited to ≤6 clients)"
  echo "  -C, --config <file>      Path to configuration file (default: mock_etcd/configuration.json)"
  echo "  -h, --help               Display this help and exit"
  echo
  echo "Examples:"
  echo "  $0 -e n_cmapss -c '0 1 2' -r 5"
  echo "  $0 -e mnist -c 6 -s -i                # Use 6 clients (0-5)"
  echo "  $0 -e mnist -c '0 1 2 3 4 5' -s -i    # Explicitly specify client IDs"
  echo "  $0 -e mnist -c 4 -d 'results/my_experiment'"
  echo "  $0 -e mnist -c '0 1 2 3' -s -f -i     # Force new data setup with IID distribution"
  echo "  $0 -e mnist -S 1                   # Run with scenario 1 (uses scenario's num_clients)"
  echo "  $0 -e mnist -S 1 -c 4               # Override scenario's client count"
  echo "  $0 -e mnist -S all                 # Run all scenarios sequentially"
  echo "  $0 -C custom_config.json           # Use a custom configuration file"
  exit 1
}

# Function to get all available scenarios
get_all_scenarios() {
  scenario_dir="mock_etcd/scenarios"
  # Use sort with -V option for natural sort of version numbers
  find "$scenario_dir" -name "scenario*.json" | sort -V
}

# Function to run a single scenario
run_single_scenario() {
  local scenario=$1
  local original_args=("${@:2}")

  echo "================================================================="
  echo "Running experiment with scenario $scenario"
  echo "================================================================="

  # Run the command with the specific scenario
  "${original_args[@]}" -S "$scenario"

  echo "Experiment with scenario $scenario completed."
  echo
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
SCENARIO="9"  # Default to scenario 9 (no disagreements)
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
    -S | --scenario )
      shift
      SCENARIO=$1
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

# Process clients parameter - convert number to client list if needed
if [ -n "$CLIENTS" ]; then
  # Check if CLIENTS is a single number (no spaces)
  if [[ "$CLIENTS" =~ ^[0-9]+$ ]]; then
    # Convert number to space-separated list (0 to n-1)
    num_clients=$CLIENTS
    CLIENTS=""
    for ((i=0; i<$num_clients; i++)); do
      if [ $i -eq 0 ]; then
        CLIENTS="$i"
      else
        CLIENTS="$CLIENTS $i"
      fi
    done
    echo "Using $num_clients clients: $CLIENTS"
  else
    echo "Using specified client IDs: $CLIENTS"
  fi
fi

# Check if we should run all scenarios
if [ "$SCENARIO" = "all" ]; then
  echo "Running all available scenarios..."

  # Get all scenarios
  scenario_files=$(get_all_scenarios)

  if [ -z "$scenario_files" ]; then
    echo "Error: No scenario files found in mock_etcd/scenarios/"
    exit 1
  fi

  # Store original command line without the scenario parameter
  original_cmd=("$0")

  # Add all arguments except the scenario
  [ -n "$EXPERIMENT" ] && original_cmd+=("-e" "$EXPERIMENT")
  [ -n "$CLIENTS" ] && original_cmd+=("-c" "$CLIENTS")
  [ -n "$ROUNDS" ] && original_cmd+=("-r" "$ROUNDS")
  [ -n "$LOCAL_EPOCHS" ] && original_cmd+=("-l" "$LOCAL_EPOCHS")
  [ -n "$BATCH_SIZE" ] && original_cmd+=("-b" "$BATCH_SIZE")
  [ -n "$SETUP_DATA" ] && original_cmd+=("-s")
  [ -n "$FORCE_SETUP" ] && original_cmd+=("-f")
  [ -n "$IID" ] && original_cmd+=("-i")
  [ -n "$RESULTS_DIR" ] && original_cmd+=("-d" "${RESULTS_DIR#* }")
  [ -n "$CONFIG_FILE" ] && original_cmd+=("-C" "$CONFIG_FILE")

  # Run each scenario
  for scenario_file in $scenario_files; do
    # Extract scenario number
    scenario_num=$(basename "$scenario_file" .json | sed 's/scenario//')
    run_single_scenario "$scenario_num" "${original_cmd[@]}"
  done

  echo "All scenarios completed."
  exit 0
fi

# Process scenario and prepare disagreements
if [ -n "$SCENARIO" ]; then
  # Determine scenario path
  if [ -f "$SCENARIO" ]; then
    SCENARIO_PATH="$SCENARIO"
  elif [ -f "mock_etcd/scenarios/scenario${SCENARIO}.json" ]; then
    SCENARIO_PATH="mock_etcd/scenarios/scenario${SCENARIO}.json"
  else
    echo "Error: Scenario file not found for scenario ${SCENARIO}"
    exit 1
  fi

  echo "Using scenario: $SCENARIO_PATH"

  # Extract scenario information and set up
  if [ -f "$SCENARIO_PATH" ]; then
    # Extract disagreements section and save to disagreements.json
    # Also get the number of clients from scenario if not explicitly set
    python3 -c "
import json
with open('$SCENARIO_PATH', 'r') as f:
    scenario = json.load(f)
with open('mock_etcd/disagreements.json', 'w') as f:
    json.dump(scenario.get('disagreements', {}), f, indent=2)

# If clients not explicitly set via command line, use scenario's num_clients
import sys
clients_arg = '$CLIENTS'
experiment_arg = '$EXPERIMENT'

if not clients_arg:
    scenario_clients = scenario.get('num_clients', 6)
    # Validate client count for n_cmapss
    if experiment_arg == 'n_cmapss' and scenario_clients > 6:
        print(f'Error: N-CMAPSS experiment cannot use more than 6 clients (scenario requests {scenario_clients})', file=sys.stderr)
        sys.exit(1)

    # Generate client list
    client_list = ' '.join(str(i) for i in range(scenario_clients))
    print(f'Using {scenario_clients} clients from scenario: {client_list}')

    # Write to a temp file for the shell to read
    with open('/tmp/fl_scenario_clients', 'w') as f:
        f.write(client_list)
else:
    print('Using explicitly specified clients')
    with open('/tmp/fl_scenario_clients', 'w') as f:
        f.write('')

print('Copied disagreements from scenario to mock_etcd/disagreements.json')
    "

    # Check if the Python script failed (e.g., due to client validation)
    if [ $? -ne 0 ]; then
      echo "Error: Failed to process scenario"
      exit 1
    fi

    # Read clients from scenario if not explicitly set
    if [ -z "$CLIENTS" ] && [ -f "/tmp/fl_scenario_clients" ]; then
      SCENARIO_CLIENTS=$(cat /tmp/fl_scenario_clients)
      if [ -n "$SCENARIO_CLIENTS" ]; then
        CLIENTS="$SCENARIO_CLIENTS"
        OVERRIDE_FLAG="--override"
      fi
      rm -f /tmp/fl_scenario_clients
    fi
  fi

  # If no custom results directory is specified, append scenario tag to the auto-generated directory
  if [ -z "$RESULTS_DIR" ]; then
    # Update the configuration to include scenario tag in directory name
    python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
if not config.get('results', {}).get('custom_dir'):
    config.setdefault('results', {})['directory_suffix'] = '_s${SCENARIO}'
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
    "
  fi
fi

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

echo "Using scenario: $SCENARIO"
echo "Parameters will override configuration in $CONFIG_FILE"
echo "Running command: $CMD"

# Create logs directory if it doesn't exist
mkdir -p logs

# Define a log file path with a timestamp
LOG_FILE="logs/experiment_$(date +%Y%m%d_%H%M%S)_${EXPERIMENT:-all}_s${SCENARIO:-none}.log"

echo "Logging output to: $LOG_FILE"

# Execute the command, redirecting stdout and stderr to tee, which writes to log file and terminal
eval $CMD 2>&1 | tee "$LOG_FILE"

# Restore the configuration file to its original state if we modified it
if [ -n "$SCENARIO" ] && [ -z "$RESULTS_DIR" ]; then
  python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
if 'directory_suffix' in config.get('results', {}):
    del config['results']['directory_suffix']
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
  "
fi

echo "Experiment completed."
