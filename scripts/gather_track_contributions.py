#!/usr/bin/env python3
"""
Script to gather track contribution PNG files from the results folder
and organize them by scenario.
"""

import os
import shutil
import re
from pathlib import Path
from collections import defaultdict

def extract_scenario_from_path(path):
    """Extract scenario number from the simulation directory name."""
    # Pattern to match scenario number from directory name like 'fl_simulation_20250615_003242_mnist_s1'
    match = re.search(r'_s(\d+)$', path.name)
    if match:
        return int(match.group(1))
    return None

def extract_dataset_from_path(path):
    """Extract dataset name from the simulation directory name."""
    # Pattern to match dataset name from directory name like 'fl_simulation_20250615_003242_mnist_s1'
    # Look for known dataset names right before _s<number>
    if 'n_cmapss' in path.name:
        return 'n_cmapss'
    elif 'mnist' in path.name:
        return 'mnist'
    else:
        # Fallback: try to extract the last word before _s<number>
        match = re.search(r'_([a-zA-Z_]+)_s\d+$', path.name)
        if match:
            return match.group(1)
    return "unknown"

def main():
    # Define paths
    results_dir = Path("results")
    output_dir = Path("results/track_contributions")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Find all track contribution PNG files
    print("Scanning for track contribution PNG files...")

    copied_files = []

    for sim_dir in results_dir.iterdir():
        if sim_dir.is_dir() and sim_dir.name.startswith("fl_simulation_"):
            sim_output_dir = sim_dir / "output"

            if sim_output_dir.exists():
                # Look for track contribution PNG files
                for png_file in sim_output_dir.glob("track_contributions_*.png"):
                    scenario_num = extract_scenario_from_path(sim_dir)
                    dataset = extract_dataset_from_path(sim_dir)

                    if scenario_num is not None:
                        # Create filename in format: s<scenario>_<experiment>_track_contributions.png
                        new_filename = f"s{scenario_num}_{dataset}_track_contributions.png"
                        destination = output_dir / new_filename

                        # Copy the file
                        shutil.copy2(png_file, destination)
                        copied_files.append(new_filename)
                        print(f"Copied: {png_file.name} -> {new_filename}")

    print(f"\nDone! All track contribution files are in: {output_dir}")
    print(f"Total files copied: {len(copied_files)}")

    # Print summary by dataset and scenario
    print(f"\nSummary:")
    mnist_files = [f for f in copied_files if "mnist" in f]
    ncmapss_files = [f for f in copied_files if "n_cmapss" in f]

    if mnist_files:
        print(f"  MNIST files: {len(mnist_files)}")
    if ncmapss_files:
        print(f"  N-CMAPSS files: {len(ncmapss_files)}")

    print(f"\nAll files are now in a single folder: {output_dir}")

if __name__ == "__main__":
    main()
