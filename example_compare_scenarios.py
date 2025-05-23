#!/usr/bin/env python3
"""
Example script showing how to use the FL Run Comparison Tool

This demonstrates common comparison scenarios you might want to run.
"""

import subprocess
import sys
from pathlib import Path

def run_comparison(run_paths, names, output_dir, description):
    """Run a comparison with the given parameters."""
    print(f"\n🔍 {description}")
    print("="*60)

    cmd = [
        "python", "compare_fl_runs.py"
    ] + run_paths + [
        "--names"
    ] + names + [
        "--output-dir", output_dir
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"✅ Comparison saved to {output_dir}/")
    else:
        print(f"❌ Comparison failed")

    return result.returncode == 0

def main():
    """Run example comparisons."""
    results_dir = Path("results")

    # Check if results directory exists
    if not results_dir.exists():
        print("❌ Results directory not found. Please run some FL experiments first.")
        return

    print("🚀 FL Run Comparison Examples")
    print("This script demonstrates common comparison scenarios.")

    # Example 1: Compare scenario 3 with different client counts
    example1_runs = [
        "results/fl_simulation_20250523_022338_mnist_s3",  # 6 clients
        "results/fl_simulation_20250523_021925_mnist_s3"   # 20 clients
    ]

    if all(Path(run).exists() for run in example1_runs):
        run_comparison(
            example1_runs,
            ["S3_6clients", "S3_20clients"],
            "comparison_s3_scaling",
            "Scenario 3: Scaling Analysis (6 vs 20 clients)"
        )

    # Example 2: Compare different scenarios with same client count
    # Look for recent runs with 6 clients
    all_runs = list(results_dir.glob("fl_simulation_*_mnist_s*"))
    recent_6client_runs = []

    for run_path in sorted(all_runs, reverse=True):
        # Try to load timing data to check client count
        timing_file = run_path / "output" / "aggregation_timing_metrics.json"
        if timing_file.exists():
            try:
                import json
                with open(timing_file) as f:
                    timing_data = json.load(f)
                    if timing_data and timing_data[0].get("num_clients") == 6:
                        recent_6client_runs.append(str(run_path))
                        if len(recent_6client_runs) >= 3:  # Limit to 3 for comparison
                            break
            except:
                continue

    if len(recent_6client_runs) >= 2:
        scenario_names = []
        for run_path in recent_6client_runs:
            # Extract scenario number
            if "_s" in run_path:
                scenario_part = run_path.split("_s")[-1]
                scenario_num = ""
                for char in scenario_part:
                    if char.isdigit():
                        scenario_num += char
                    else:
                        break
                scenario_names.append(f"Scenario_{scenario_num}")
            else:
                scenario_names.append("Unknown")

        run_comparison(
            recent_6client_runs,
            scenario_names,
            "comparison_scenarios_6clients",
            "Scenario Comparison: Different scenarios with 6 clients"
        )

    # Example 3: Compare scenario 16 vs 17 (if they exist)
    s16_runs = list(results_dir.glob("fl_simulation_*_mnist_s16"))
    s17_runs = list(results_dir.glob("fl_simulation_*_mnist_s17"))

    if s16_runs and s17_runs:
        run_comparison(
            [str(s16_runs[0]), str(s17_runs[0])],
            ["S16_Ring_Disagreements", "S17_Baseline"],
            "comparison_s16_vs_s17",
            "Scenario 16 vs 17: Disagreement Ring vs Baseline"
        )

    print("\n✨ Examples completed!")
    print("You can also run custom comparisons like:")
    print("python compare_fl_runs.py results/run1 results/run2 --names 'Run1' 'Run2' --output-dir my_comparison")

if __name__ == "__main__":
    main()
