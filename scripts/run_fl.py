#!/usr/bin/env python3
"""
Federated Learning Experiment Runner

This script runs federated learning experiments with support for different
scenarios, experiment types, and configuration overrides.
"""

import argparse
import json
import os
import subprocess
import sys
import glob
from datetime import datetime
from pathlib import Path


def usage():
    """Print usage information."""
    print("""Usage: run_fl.py [options]

Options:
  -e, --experiment <n>     Experiment type (n_cmapss or mnist).
  -c, --clients <ids>      Number of clients or a list of IDs (e.g., 6 or '0 1 3 5').
                           If not specified, uses 'num_clients' from the scenario file.
  -r, --rounds <num>       Number of FL rounds (default: 3).
  -l, --local-epochs <num> Number of local training epochs (default: 5).
  -b, --batch-size <num>   Batch size (default: 64).
  -s, --setup-data         Setup data (for MNIST only).
  -f, --force-setup        Force data setup even if it exists (for MNIST only).
  -i, --iid                Use IID data distribution (for MNIST only).
  -d, --results-dir <dir>  Custom results directory (default: auto-generated).
  -S, --scenario <num>     Scenario to run by number or 'all' (default: 0).
                           Scenarios can define 'num_clients' (N-CMAPSS is limited to <= 6).
  -C, --config <file>      Path to configuration file (default: mock_etcd/configuration.json).
  --no-viz                 Skip automatic track visualization generation.
  --verbose-plots          Generate all plots (default: only last round track metrics + track contributions).
  -h, --help               Display this help and exit.

Examples:
  run_fl.py -e n_cmapss -c 3 -r 5
  run_fl.py -e mnist -c 6 -s -i
  run_fl.py -e mnist -c 4 -d 'results/my_experiment'
  run_fl.py -e mnist -c 4 -s -f -i
  run_fl.py -e mnist -S 1
  run_fl.py -e mnist -S 1 -c 4
  run_fl.py -e mnist -S all
  run_fl.py -C custom_config.json
  run_fl.py -e mnist -S 1 --no-viz
  run_fl.py -e n_cmapss -c 3 -r 5 --verbose-plots

Note: By default, only track metrics comparison plots for the last round are generated
      to improve performance. Use --verbose-plots to generate all plots for all rounds.
      Track contributions visualization is automatically generated after each
      experiment completion and saved to the simulation's output/ directory.
""")


def get_all_scenarios():
    """Get all available scenario files sorted naturally."""
    scenario_dir = "mock_etcd/scenarios"
    scenario_files = glob.glob(os.path.join(scenario_dir, "scenario*.json"))
    # Sort naturally by scenario number
    return sorted(scenario_files, key=lambda f: int(os.path.basename(f).replace("scenario", "").replace(".json", "")))


def run_single_scenario(scenario, original_args):
    """Run a single scenario with the given arguments."""
    print("=" * 65)
    print(f"Running experiment with scenario {scenario}")
    print("=" * 65)

    # Create new args with this specific scenario
    new_args = original_args.copy()
    new_args.scenario = scenario

    # Run the experiment
    run_experiment(new_args)

    print(f"Experiment with scenario {scenario} completed.")
    print()


def process_clients_parameter(clients_str):
    """Process the clients parameter, converting number to client list if needed."""
    if not clients_str:
        return ""

    # Check if it's a single number (no spaces)
    if clients_str.strip().isdigit():
        num_clients = int(clients_str.strip())
        clients_list = [str(i) for i in range(num_clients)]
        clients_result = " ".join(clients_list)
        print(f"Using {num_clients} clients: {clients_result}")
        return clients_result
    else:
        print(f"Using specified client IDs: {clients_str}")
        return clients_str


def setup_scenario(scenario, config_file, experiment_type, clients):
    """Setup scenario by extracting disagreements and determining clients."""
    # Determine scenario path
    if os.path.isfile(scenario):
        scenario_path = scenario
    elif os.path.isfile(f"mock_etcd/scenarios/scenario{scenario}.json"):
        scenario_path = f"mock_etcd/scenarios/scenario{scenario}.json"
    else:
        print(f"Error: Scenario file not found for scenario {scenario}")
        sys.exit(1)

    print(f"Using scenario: {scenario_path}")

    # Load scenario and extract disagreements
    try:
        with open(scenario_path, 'r') as f:
            scenario_data = json.load(f)

        # Save disagreements to mock_etcd/disagreements.json
        with open('mock_etcd/disagreements.json', 'w') as f:
            json.dump(scenario_data.get('disagreements', {}), f, indent=2)

        print('Copied disagreements from scenario to mock_etcd/disagreements.json')

        # If clients not explicitly set, use scenario's num_clients
        if not clients:
            scenario_clients = scenario_data.get('num_clients', 6)

            # Validate client count for n_cmapss
            if experiment_type == 'n_cmapss' and scenario_clients > 6:
                print(f'Error: N-CMAPSS experiment cannot use more than 6 clients (scenario requests {scenario_clients})', file=sys.stderr)
                sys.exit(1)

            # Generate client list
            client_list = ' '.join(str(i) for i in range(scenario_clients))
            print(f'Using {scenario_clients} clients from scenario: {client_list}')
            clients = client_list
        else:
            print('Using explicitly specified clients')

        # If no custom results directory specified, add scenario tag
        # This is handled by modifying the config file temporarily
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            if not config.get('results', {}).get('custom_dir'):
                config.setdefault('results', {})['directory_suffix'] = f'_s{scenario}'

                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update config file: {e}")

        return clients, scenario_path

    except Exception as e:
        print(f"Error: Failed to process scenario: {e}")
        sys.exit(1)


def restore_config(config_file, scenario, results_dir):
    """Restore the configuration file to its original state."""
    if scenario and not results_dir:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            if 'directory_suffix' in config.get('results', {}):
                del config['results']['directory_suffix']

                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not restore config file: {e}")


def find_latest_simulation_dir(experiment_type=None, scenario=None):
    """Find the most recent FL simulation directory."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None

    # Build pattern to match simulation directories
    if experiment_type and scenario and scenario != "0":
        pattern = f"fl_simulation_*_{experiment_type}_s{scenario}"
    elif experiment_type:
        pattern = f"fl_simulation_*_{experiment_type}*"
    else:
        pattern = "fl_simulation_*"

    # Find all matching directories
    sim_dirs = list(results_dir.glob(pattern))

    if not sim_dirs:
        # Try without scenario suffix
        sim_dirs = list(results_dir.glob("fl_simulation_*"))

    if not sim_dirs:
        return None

    # Return the most recent one (based on directory name timestamp)
    return str(max(sim_dirs, key=lambda x: x.stat().st_mtime))


def run_track_visualization(simulation_path, fl_rounds=None):
    """Run the track contributions visualization for a completed experiment."""
    if not simulation_path or not os.path.exists(simulation_path):
        print("Warning: No valid simulation directory found for visualization")
        return

    # Check if the simulation has track metadata
    model_storage = Path(simulation_path) / "model_storage"
    if not model_storage.exists():
        print("Warning: No model_storage directory found, skipping visualization")
        return

    # Check for track metadata in any round
    track_files = list(model_storage.glob("round_*/tracks/track_metadata.json"))
    if not track_files:
        print("Warning: No track metadata found, skipping visualization")
        return

    print("\n" + "="*50)
    print("Generating track contributions visualization...")
    print("="*50)

    try:
        # Run the visualization script
        viz_script = "scripts/visualize_track_contributions.py"
        cmd = ["python", viz_script, simulation_path]

        # Add rounds parameter if available
        if fl_rounds:
            cmd.extend(["--rounds", str(fl_rounds)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("Track visualization completed successfully!")
            # Print the last line which should contain the save path
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "Visualization saved to:" in line:
                    print(line)
        else:
            print(f"Warning: Visualization failed with error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("Warning: Visualization timed out")
    except Exception as e:
        print(f"Warning: Could not run visualization: {e}")


def run_experiment(args):
    """Run a federated learning experiment with the given arguments."""
    # Process clients parameter
    clients = process_clients_parameter(args.clients) if args.clients else ""

    # Handle 'all' scenarios
    if args.scenario == "all":
        print("Running all available scenarios...")

        scenario_files = get_all_scenarios()
        if not scenario_files:
            print("Error: No scenario files found in mock_etcd/scenarios/")
            sys.exit(1)

        # Run each scenario
        for scenario_file in scenario_files:
            scenario_num = os.path.basename(scenario_file).replace("scenario", "").replace(".json", "")
            run_single_scenario(scenario_num, args)

        print("All scenarios completed.")
        return

    # Setup scenario if specified
    if args.scenario:
        clients, scenario_path = setup_scenario(args.scenario, args.config, args.experiment, clients)

    # Build the command to run fl_orchestrator.py
    cmd = ["uv", "run", "fl_orchestrator.py", "--config", args.config]

    # Add override flag if any parameters are specified
    if any([args.experiment, clients, args.rounds, args.local_epochs, args.batch_size,
            args.setup_data, args.force_setup, args.iid, args.results_dir, args.verbose_plots]):
        cmd.append("--override")

    # Add override arguments
    if args.experiment:
        cmd.extend(["--experiment", args.experiment])

    if clients:
        # Split the clients string and add each client ID as a separate argument
        client_ids = clients.split()
        cmd.extend(["--clients"] + client_ids)

    if args.rounds:
        cmd.extend(["--fl_rounds", str(args.rounds)])

    if args.local_epochs:
        cmd.extend(["--local_epochs", str(args.local_epochs)])

    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])

    if args.setup_data:
        cmd.append("--setup_data")

    if args.force_setup:
        cmd.append("--force_setup_data")

    if args.iid:
        cmd.append("--iid")

    if args.results_dir:
        cmd.extend(["--results_dir", args.results_dir])

    if args.verbose_plots:
        cmd.append("--verbose_plots")

    # Print experiment details
    if args.experiment:
        print(f"Running {args.experiment} federated learning experiment")

    if clients:
        print(f"Using clients: {clients}")

    print(f"Using scenario: {args.scenario}")
    print(f"Parameters will override configuration in {args.config}")
    print(f"Running command: {' '.join(cmd)}")

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Define log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment or "all"
    scenario_name = args.scenario or "none"
    log_file = f"logs/experiment_{timestamp}_{experiment_name}_s{scenario_name}.log"

    print(f"Logging output to: {log_file}")

    # Execute the command and log output
    try:
        with open(log_file, 'w') as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Print to console
                log.write(line)      # Write to log file
                log.flush()          # Ensure immediate write

            process.wait()

            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}")
                sys.exit(process.returncode)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running command: {e}")
        sys.exit(1)

    finally:
        # Restore the configuration file
        restore_config(args.config, args.scenario, args.results_dir)

    # Generate track visualization for completed experiment
    if not args.no_viz:
        simulation_dir = find_latest_simulation_dir(args.experiment, args.scenario)
        if simulation_dir:
            run_track_visualization(simulation_dir, args.rounds)
    else:
        print("Skipping track visualization (--no-viz specified)")

    print("Experiment completed.")


def main():
    """Main function to parse arguments and run the experiment."""
    parser = argparse.ArgumentParser(description='Run federated learning experiments',
                                   add_help=False)  # We handle help manually

    parser.add_argument('-e', '--experiment', type=str,
                       help='Experiment type (n_cmapss or mnist)')
    parser.add_argument('-c', '--clients', type=str,
                       help='Client IDs (e.g., "0 1 2 3 4 5") or number of clients (e.g., 6)')
    parser.add_argument('-r', '--rounds', type=int,
                       help='Number of FL rounds (default: 3)')
    parser.add_argument('-l', '--local-epochs', type=int, dest='local_epochs',
                       help='Number of local training epochs (default: 5)')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size',
                       help='Batch size (default: 64)')
    parser.add_argument('-s', '--setup-data', action='store_true', dest='setup_data',
                       help='Setup data (for MNIST only)')
    parser.add_argument('-f', '--force-setup', action='store_true', dest='force_setup',
                       help='Force data setup even if it exists (for MNIST only)')
    parser.add_argument('-i', '--iid', action='store_true',
                       help='Use IID data distribution (for MNIST only)')
    parser.add_argument('-d', '--results-dir', type=str, dest='results_dir',
                       help='Custom results directory (default: auto-generated)')
    parser.add_argument('-S', '--scenario', type=str, default="0",
                       help='Scenario number to run (default: 0 - no disagreements) or "all"')
    parser.add_argument('-C', '--config', type=str, default="mock_etcd/configuration.json",
                       help='Path to configuration file (default: mock_etcd/configuration.json)')
    parser.add_argument('--no-viz', action='store_true', dest='no_viz',
                       help='Skip automatic track visualization generation')
    parser.add_argument('--verbose-plots', action='store_true', dest='verbose_plots',
                       help='Generate all plots (default: only last round track metrics + track contributions)')
    parser.add_argument('-h', '--help', action='store_true',
                       help='Display this help and exit')

    args = parser.parse_args()

    if args.help:
        usage()
        sys.exit(0)

    # Change to script directory if it is being called from elsewhere
    # This ensures relative paths work correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up one level from scripts/
    os.chdir(parent_dir)

    run_experiment(args)


if __name__ == "__main__":
    main()
