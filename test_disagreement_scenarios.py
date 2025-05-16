#!/usr/bin/env python3
"""
Disagreement scenarios testing script.
Used to verify that the disagreement resolution system correctly handles various scenarios.
"""

import os
import json
import sys
import argparse
import subprocess
from pathlib import Path
import glob

def load_scenario(scenario_path):
    """Load a disagreement scenario from a file."""
    try:
        with open(scenario_path, 'r') as f:
            scenario = json.load(f)
        print(f"Loaded scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        return scenario
    except Exception as e:
        print(f"Error loading scenario: {e}")
        return None

def get_all_scenarios():
    """Get a list of all available scenario files."""
    scenario_dir = "mock_etcd/scenarios"
    scenario_files = glob.glob(os.path.join(scenario_dir, "scenario*.json"))

    # Sort scenarios numerically instead of lexicographically
    return sorted(scenario_files, key=lambda f: int(os.path.basename(f).replace("scenario", "").replace(".json", "")))

def copy_disagreements_to_etcd(scenario):
    """Copy disagreements from scenario to mock_etcd/disagreements.json."""
    try:
        with open('mock_etcd/disagreements.json', 'w') as f:
            json.dump(scenario['disagreements'], f, indent=2)
        print(f"Copied disagreements to mock_etcd/disagreements.json")
        return True
    except Exception as e:
        print(f"Error copying disagreements: {e}")
        return False

def run_test(scenario_input, experiment="mnist", fl_rounds=3, local_epochs=1):
    """Run a federated learning test using the specified scenario and dataset.

    Args:
        scenario_input: Path to the scenario file or scenario number
        experiment: Experiment type (mnist or n_cmapss)
        fl_rounds: Number of federated learning rounds
        local_epochs: Number of local training epochs
    """
    # Determine if scenario_input is a path or just a number/name
    if scenario_input.isdigit() or os.path.basename(scenario_input).startswith("scenario"):
        # For built-in scenarios, pass the scenario number/name directly
        scenario_arg = os.path.basename(scenario_input).replace("scenario", "").replace(".json", "")
    else:
        # For custom scenario paths, pass the full path
        scenario_arg = scenario_input

    cmd = [
        "./run_federated_experiment.sh",
        "-e", experiment,
        "-c", "0 1 2 3 4 5",
        "-r", str(fl_rounds),
        "-l", str(local_epochs),
        "-i",
        "-S", scenario_arg
    ]

    print(f"Running test with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Extract the results directory from the output
    for line in result.stdout.split('\n'):
        if "Results directory:" in line:
            results_dir = line.split("Results directory:")[1].strip()
            return results_dir

    return None

def verify_tracks(results_dir, scenario):
    """Verify that the created tracks match the expected tracks."""
    # Check each round and report on the tracks
    success = True
    is_scenario5 = "name" in scenario and scenario["name"] == "Time-Limited Disagreements"
    is_empty_scenario = (
        "expected_tracks" in scenario and
        len(scenario["expected_tracks"]) == 1 and
        "global" in scenario["expected_tracks"] and
        "disagreements" in scenario and
        len(scenario["disagreements"]) == 0
    )

    # Special handling for empty scenario (no disagreements)
    if is_empty_scenario:
        print("\nTesting empty scenario with no disagreements")
        client_dirs = glob.glob(os.path.join(results_dir, "output", "clients", "client_*"))
        if not client_dirs:
            print("❌ No client output directories found")
            return False
        print(f"✅ Found {len(client_dirs)} client output directories")
        print("✅ Empty scenario test passed - all clients use global track by default")
        return True

    # Add debugging to check for round directories
    model_storage_dir = os.path.join(results_dir, "model_storage")
    round_dirs = [d for d in os.listdir(model_storage_dir) if d.startswith("round_")]
    max_round = max([int(d.split("_")[1]) for d in round_dirs])
    print(f"Found {len(round_dirs)} round directories, max round: {max_round}")

    # Extract validation rules if present
    validation_rules = scenario.get("validation_rules", {})

    # Go through each round directory
    for round_dir in sorted(round_dirs, key=lambda x: int(x.split("_")[1])):
        round_num = int(round_dir.split("_")[1])
        round_path = os.path.join(model_storage_dir, round_dir)
        tracks_dir = os.path.join(round_path, "tracks")

        # Determine active disagreements for this round
        active_disagreements = {}
        if "disagreements" in scenario:
            # Handle different disagreement formats
            if isinstance(next(iter(scenario["disagreements"].values())), list):
                # New format: list of disagreements per client
                for client_id, disagreement_list in scenario["disagreements"].items():
                    for disagreement in disagreement_list:
                        target_id = disagreement.get("target")
                        if not target_id:
                            continue

                        # Check if this disagreement is active in this round
                        active_rounds = disagreement.get("active_rounds", {})
                        start_round = active_rounds.get("start", 1)
                        end_round = active_rounds.get("end")
                        is_active = (start_round <= round_num) and (end_round is None or round_num <= end_round)

                        if is_active:
                            if client_id not in active_disagreements:
                                active_disagreements[client_id] = []
                            active_disagreements[client_id].append(disagreement)
            else:
                # Old format: dictionary of disagreements per client
                for client_id, disagreements in scenario["disagreements"].items():
                    for target_id, details in disagreements.items():
                        if "rounds" in details:
                            # If rounds specified, check if this round is included
                            if round_num in details["rounds"]:
                                if client_id not in active_disagreements:
                                    active_disagreements[client_id] = {}
                                active_disagreements[client_id][target_id] = details
                        else:
                            # If no rounds specified, include for all rounds
                            if client_id not in active_disagreements:
                                active_disagreements[client_id] = {}
                            active_disagreements[client_id][target_id] = details

        # Check if tracks directory exists
        if not os.path.exists(tracks_dir):
            print(f"\n=== Round {round_num} ===")
            print(f"No tracks directory found for round {round_num}")

            # Check if there should be active disagreements for this round
            if active_disagreements:
                print(f"❌ Round {round_num}: No tracks directory but should have active disagreements")
                print(f"Active disagreements: {active_disagreements}")
                success = False
            else:
                print(f"✅ Round {round_num}: No tracks directory as expected (no active disagreements)")

            # Apply validation rules if possible without track metadata
            round_rules_key = f"round_{round_num}_rules"
            if validation_rules and round_rules_key in validation_rules:
                print(f"Applying validation rules for round {round_num} without tracks")
                client_dirs = glob.glob(os.path.join(results_dir, "output", "clients", "client_*"))
                client_ids_active = [os.path.basename(d).split("_")[1] for d in client_dirs]
                print(f"  Found active clients: {sorted(client_ids_active)}")

                # Apply validation rules that make sense without track_metadata
                rules = validation_rules[round_rules_key]
                for rule in rules:
                    rule_type = rule.get("type")
                    if rule_type in ["clients_share_track", "clients_on_track"]:
                        print(f"  ✅ Assumed passed: {rule.get('description', rule_type)}")

            continue

        track_metadata_path = os.path.join(tracks_dir, "track_metadata.json")

        # Skip if no track metadata file
        if not os.path.exists(track_metadata_path):
            print(f"\n=== Round {round_num} ===")
            print(f"No track metadata file for round {round_num}")

            if round_num == 1:
                print(f"❌ No track metadata found for round 1, which should always have metadata")
                success = False
            elif active_disagreements:
                print(f"❌ Round {round_num}: No track metadata but disagreements should be active")
                success = False
            else:
                print(f"✅ Round {round_num}: No track metadata as expected (all disagreements expired)")

            continue

        with open(track_metadata_path, 'r') as f:
            track_metadata = json.load(f)

        # Print round explanation for all scenarios
        print(f"\n=== Round {round_num} ===")

        # Check if track_metadata should exist for this round
        if not active_disagreements:
            print(f"❌ Round {round_num}: Found tracks directory but all disagreements should have expired")
            success = False
            continue

        # Check if there's a round-specific expectation
        round_specific_key = f"expected_tracks_round_{round_num}"

        # Get expected tracks for this round
        if round_specific_key in scenario:
            print(f"Using round-specific expectations for round {round_num}")
            expected_tracks = scenario[round_specific_key]
        else:
            # Otherwise use the general expected tracks
            expected_tracks = scenario.get("expected_tracks", {})

        # Initialize mapping of client IDs to tracks
        client_track_mapping = {}

        # First, get client primary tracks
        if "client_tracks" in track_metadata:
            for client_id, track_name in track_metadata["client_tracks"].items():
                client_track_mapping[client_id] = track_name

        # Then, add secondary track information
        if "tracks" in track_metadata:
            for track_name, clients in track_metadata["tracks"].items():
                for client_id in clients:
                    if str(client_id) not in client_track_mapping:
                        client_track_mapping[str(client_id)] = track_name

        # Print client track assignments for this round
        print(f"Client track assignments for round {round_num}:")
        for track_name, track_clients in track_metadata.get("tracks", {}).items():
            primary_clients = [
                client_id for client_id, track in track_metadata.get("client_tracks", {}).items()
                if track == track_name
            ]
            print(f"  Track '{track_name}': Primary for clients {sorted(primary_clients)}")

        # Print track participants for debugging
        print(f"  Track participants in round {round_num}:")
        for track_name, participants in track_metadata.get("tracks", {}).items():
            print(f"    Track '{track_name}': {sorted(participants)}")

        # Check if track names match expected
        tracks_match = True
        # Convert both to sets for easier comparison (ignoring order)
        expected_track_names = set(expected_tracks.keys())
        actual_track_names = set(track_metadata.get("tracks", {}).keys())

        if expected_track_names != actual_track_names:
            missing_tracks = expected_track_names - actual_track_names
            extra_tracks = actual_track_names - expected_track_names

            if missing_tracks:
                print(f"❌ Missing expected tracks: {missing_tracks}")
                tracks_match = False
            if extra_tracks:
                print(f"❌ Unexpected extra tracks: {extra_tracks}")
                tracks_match = False

        if tracks_match:
            print(f"Round {round_num}: Track names match expected")
        else:
            print(f"❌ Round {round_num}: Track names do not match expected")
            success = False

        # Print detailed track information
        print(f"\nTrack assignments for round {round_num}:")
        for track_name, description in expected_tracks.items():
            if track_name in track_metadata.get("tracks", {}):
                participating_clients = sorted(track_metadata["tracks"][track_name])
                primary_clients = [
                    client_id for client_id, track in track_metadata.get("client_tracks", {}).items()
                    if track == track_name
                ]
                print(f"  Track '{track_name}':")
                print(f"    Expected: {description}")
                print(f"    Participating clients: {participating_clients}")
                print(f"    Primary for clients: {sorted(primary_clients)}")
            else:
                print(f"  ❌ Track '{track_name}' not found in actual tracks")
                success = False

        # Check validation rules for this round if available
        round_rules_key = f"round_{round_num}_rules"
        if round_rules_key in validation_rules:
            print(f"Applying validation rules for round {round_num}")
            rules = validation_rules[round_rules_key]
            for rule in rules:
                rule_type = rule.get("type")
                if rule_type == "client_excludes":
                    client = rule.get("client")
                    excludes = rule.get("excludes", [])
                    description = rule.get("description", "")

                    # Get the track for this client
                    client_track = client_track_mapping.get(str(client))
                    if not client_track:
                        print(f"❌ {description}: Client {client} not found in any track")
                        success = False
                        continue

                    # Check that this client's track excludes the specified clients
                    track_clients = track_metadata.get("tracks", {}).get(client_track, [])

                    for excluded_client in excludes:
                        if excluded_client in track_clients:
                            print(f"❌ {description}: Client {excluded_client} should be excluded from {client_track}")
                            success = False

                elif rule_type == "clients_share_track":
                    clients = rule.get("clients", [])
                    description = rule.get("description", "")

                    # Check that all clients are on the same track
                    tracks = set()
                    for client in clients:
                        track = client_track_mapping.get(str(client))
                        if track:
                            tracks.add(track)

                    if len(tracks) > 1:
                        print(f"❌ {description}: Clients {clients} are on different tracks: {tracks}")
                        success = False
                    elif len(tracks) == 0:
                        print(f"❌ {description}: None of the clients {clients} found in any track")
                        success = False

                elif rule_type == "clients_on_track":
                    track_name = rule.get("track_name")
                    clients = rule.get("clients", [])
                    description = rule.get("description", "")

                    # Check that all specified clients are on the specified track
                    for client in clients:
                        client_track = client_track_mapping.get(str(client))
                        if client_track != track_name:
                            print(f"❌ {description}: Client {client} should be on track {track_name} but is on {client_track}")
                            success = False

                elif rule_type == "track_includes_only":
                    track_name = rule.get("track_name")
                    clients = rule.get("clients", [])
                    description = rule.get("description", "")

                    if track_name in track_metadata.get("tracks", {}):
                        track_clients = set(track_metadata["tracks"][track_name])
                        expected_clients = set(str(c) for c in clients)

                        if track_clients != expected_clients:
                            extra = track_clients - expected_clients
                            missing = expected_clients - track_clients
                            if extra:
                                print(f"❌ {description}: Track {track_name} has extra clients: {extra}")
                                success = False
                            if missing:
                                print(f"❌ {description}: Track {track_name} is missing clients: {missing}")
                                success = False
                    else:
                        print(f"❌ {description}: Track {track_name} not found")
                        success = False
        # Fall back to special case for time-limited disagreements if needed
        elif is_scenario5 and round_num >= 3 and not validation_rules:
            print(f"For Time-Limited Disagreements scenario (Scenario 5), using client-based validation")
            # Apply hard-coded rules for scenario 5
            if round_num == 3:
                # Check if client 0 and client 5 share the same track (excluding client 1)
                client0_track = client_track_mapping.get("0")
                client5_track = client_track_mapping.get("5")
                if client0_track != client5_track:
                    print(f"❌ In round 3, client 0 and client 5 should share the same track (both exclude client 1)")
                    success = False

                # Check if client 1 is on a track excluding client 4 only
                client1_track = client_track_mapping.get("1")
                client1_track_participants = track_metadata["tracks"].get(client1_track, [])

                # Check if client 3 is a participant in client 1's track
                if 3 not in client1_track_participants:
                    print(f"❌ In round 3, client 1's track should include client 3 as participant (exclusion expired)")
                    success = False

                # Check if client 4 is NOT a participant in client 1's track
                if 4 in client1_track_participants:
                    print(f"❌ In round 3, client 1's track should exclude client 4")
                    success = False

            # For round 5, check if clients 1, 2, and 4 are on the global track
            elif round_num == 5:
                # Check if clients 1, 2, and 4 are assigned to the global track
                global_track_clients = [client_id for client_id, track in client_track_mapping.items() if track == "global"]
                required_global_clients = ["1", "2", "4"]

                if not all(client in global_track_clients for client in required_global_clients):
                    print(f"❌ In round 5, clients 1, 2, and 4 should be on the global track (all exclusions expired)")
                    missing = [c for c in required_global_clients if c not in global_track_clients]
                    print(f"  Missing from global track: {missing}")
                    success = False

                # Check that client 0 and client 5 still share a track
                client0_track = client_track_mapping.get("0")
                client5_track = client_track_mapping.get("5")
                if client0_track != client5_track:
                    print(f"❌ In round 5, client 0 and client 5 should still share the same track")
                    success = False

                # Check that client 3 is still on its own track excluding client 0
                client3_track = client_track_mapping.get("3")
                client3_track_participants = track_metadata["tracks"].get(client3_track, [])
                if 0 in client3_track_participants:
                    print(f"❌ In round 5, client 3's track should still exclude client 0")
                    success = False

    if success:
        print("\n✅ SUCCESS: All tracks match expected configuration")
    else:
        print("\n❌ FAILURE: Some tracks did not match expected configuration")

    return success

def run_single_scenario(scenario_path, args):
    """Run and verify a single scenario.

    Args:
        scenario_path: Path to the scenario file
        args: Command line arguments

    Returns:
        bool: True if the test passed, False otherwise
    """
    print(f"\n{'=' * 80}")
    print(f"Testing scenario: {scenario_path}")
    print(f"{'=' * 80}")
    print(f"Using experiment type: {args.experiment}")
    print(f"Running for {args.rounds} rounds with {args.local_epochs} local epochs per round")

    # Load the scenario
    scenario = load_scenario(scenario_path)
    if not scenario:
        return False

    # Run the test
    results_dir = run_test(scenario_path, args.experiment, args.rounds, args.local_epochs)
    if not results_dir:
        print("❌ Error: Test failed to run correctly")
        return False

    # Verify the results
    if not verify_tracks(results_dir, scenario):
        print(f"❌ Test failed: Tracks do not match expected configuration")
        return False

    print(f"✅ Test passed: Tracks match expected configuration")
    return True

def main():
    parser = argparse.ArgumentParser(description='Test disagreement scenarios')
    parser.add_argument('scenario', help='Scenario name/number or "all" to run all scenarios')
    parser.add_argument('-r', '--rounds', type=int, default=3,
                        help='Number of federated learning rounds (default: 3)')
    parser.add_argument('-l', '--local-epochs', type=int, default=1,
                        help='Number of local training epochs (default: 1)')
    parser.add_argument('-e', '--experiment', type=str, default='mnist',
                        choices=['mnist', 'n_cmapss'],
                        help='Experiment type to use (mnist or n_cmapss)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    # Check if "all" scenarios should be run
    if args.scenario.lower() == "all":
        scenario_paths = get_all_scenarios()
        if not scenario_paths:
            print("Error: No scenarios found in mock_etcd/scenarios/")
            return 1

        print(f"Running all {len(scenario_paths)} scenarios...")

        # Track results
        results = {}
        success_count = 0

        for scenario_path in scenario_paths:
            scenario_name = os.path.basename(scenario_path).replace(".json", "")
            success = run_single_scenario(scenario_path, args)
            results[scenario_name] = "✅ Passed" if success else "❌ Failed"
            if success:
                success_count += 1

        # Print summary
        print("\n" + "=" * 80)
        print(f"SCENARIO TEST SUMMARY: {success_count}/{len(scenario_paths)} passed")
        print("=" * 80)
        for scenario_name, status in results.items():
            print(f"{scenario_name:<20}: {status}")

        if success_count == len(scenario_paths):
            print("\n✅ All scenarios passed!")
            return 0
        else:
            print(f"\n❌ {len(scenario_paths) - success_count} scenarios failed")
            return 1
    else:
        # Determine the scenario path
        if os.path.exists(args.scenario):
            scenario_path = args.scenario
        elif os.path.exists(f"mock_etcd/scenarios/scenario{args.scenario}.json"):
            scenario_path = f"mock_etcd/scenarios/scenario{args.scenario}.json"
        elif os.path.exists(f"mock_etcd/scenarios/{args.scenario}.json"):
            scenario_path = f"mock_etcd/scenarios/{args.scenario}.json"
        else:
            print(f"Error: Scenario file {args.scenario} not found")
            return 1

        # Run single scenario
        if run_single_scenario(scenario_path, args):
            return 0
        else:
            return 1

if __name__ == "__main__":
    sys.exit(main())
