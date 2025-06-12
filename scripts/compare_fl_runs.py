#!/usr/bin/env python3
"""
Federated Learning Run Comparison Tool

This script compares multiple federated learning simulation runs,
analyzing performance metrics, timing data, and scenario characteristics.
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class FLRunComparator:
    def __init__(self):
        self.runs = {}

    def load_run(self, run_path, run_name=None):
        """Load a federated learning run from results directory."""
        run_path = Path(run_path)

        if run_name is None:
            run_name = run_path.name

        print(f"Loading run: {run_name}")

        # Load main results
        fl_results_path = run_path / "output" / "fl_results.json"
        timing_metrics_path = run_path / "output" / "aggregation_timing_metrics.json"

        run_data = {
            "name": run_name,
            "path": str(run_path),
            "loaded_at": datetime.now()
        }

        # Load FL results
        if fl_results_path.exists():
            with open(fl_results_path, 'r') as f:
                fl_results = json.load(f)
                run_data["fl_results"] = fl_results
                run_data["experiment_type"] = fl_results.get("experiment_type", "unknown")
        else:
            print(f"Warning: No fl_results.json found in {run_path}")
            return None

        # Load timing metrics
        if timing_metrics_path.exists():
            with open(timing_metrics_path, 'r') as f:
                timing_data = json.load(f)
                run_data["timing_metrics"] = timing_data
        else:
            print(f"Warning: No timing metrics found in {run_path}")
            run_data["timing_metrics"] = []

        # Extract scenario info from directory name
        dir_name = run_path.name
        if "_s" in dir_name:
            scenario_part = dir_name.split("_s")[-1]
            scenario_num = ""
            for char in scenario_part:
                if char.isdigit():
                    scenario_num += char
                else:
                    break
            run_data["scenario"] = int(scenario_num) if scenario_num else None
        else:
            run_data["scenario"] = None

        # Determine number of clients from timing data
        if run_data["timing_metrics"]:
            run_data["num_clients"] = run_data["timing_metrics"][0].get("num_clients", "unknown")
        else:
            run_data["num_clients"] = "unknown"

        # Extract basic metrics
        self._extract_summary_metrics(run_data)

        self.runs[run_name] = run_data
        print(f"✓ Loaded run: {run_name} (Scenario {run_data['scenario']}, {run_data['num_clients']} clients)")

        return run_data

    def _extract_summary_metrics(self, run_data):
        """Extract summary metrics from the run data."""
        fl_results = run_data.get("fl_results", {})
        timing_metrics = run_data.get("timing_metrics", [])

        # Performance metrics
        rounds_data = fl_results.get("rounds", [])
        if rounds_data:
            final_round = rounds_data[-1]
            run_data["final_accuracy"] = final_round.get("test_accuracy")
            run_data["final_loss"] = final_round.get("test_loss")
            run_data["final_precision"] = final_round.get("mean_precision")
            run_data["final_recall"] = final_round.get("mean_recall")
            run_data["final_f1"] = final_round.get("mean_f1")
            run_data["total_rounds"] = len([r for r in rounds_data if r["round"] > 0])

        # Timing metrics summary
        if timing_metrics:
            total_times = [entry["total_aggregation_time_seconds"] for entry in timing_metrics]
            aggregation_times = [entry["aggregation_time_seconds"] for entry in timing_metrics]
            resolution_times = [entry["resolution_time_seconds"] * 1000 for entry in timing_metrics]  # in ms
            has_disagreements = [entry["has_disagreements"] for entry in timing_metrics]

            run_data["avg_total_time"] = np.mean(total_times)
            run_data["avg_aggregation_time"] = np.mean(aggregation_times)
            run_data["avg_resolution_time_ms"] = np.mean(resolution_times)
            run_data["rounds_with_disagreements"] = sum(has_disagreements)
            run_data["total_timing_rounds"] = len(timing_metrics)

            # Calculate overhead
            with_disag_times = [t for t, has_disag in zip(total_times, has_disagreements) if has_disag]
            without_disag_times = [t for t, has_disag in zip(total_times, has_disagreements) if not has_disag]

            if with_disag_times and without_disag_times:
                avg_with = np.mean(with_disag_times)
                avg_without = np.mean(without_disag_times)
                run_data["disagreement_overhead_pct"] = ((avg_with - avg_without) / avg_without) * 100
            else:
                run_data["disagreement_overhead_pct"] = None

    def compare_performance(self, save_plots=True, output_dir=None):
        """Compare performance metrics across runs."""
        if len(self.runs) < 2:
            print("Need at least 2 runs to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        # Performance comparison
        plt.figure(figsize=(15, 10))

        # Prepare data for comparison
        run_names = list(self.runs.keys())
        metrics = ['final_accuracy', 'final_precision', 'final_recall', 'final_f1']
        metric_titles = ['Final Accuracy', 'Final Precision', 'Final Recall', 'Final F1 Score']

        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            plt.subplot(2, 2, i+1)

            values = []
            labels = []
            colors = []

            for run_name, run_data in self.runs.items():
                value = run_data.get(metric)
                if value is not None:
                    values.append(value)
                    scenario = run_data.get('scenario', 'Unknown')
                    clients = run_data.get('num_clients', 'Unknown')
                    labels.append(f"S{scenario}\n({clients} clients)")

                    # Color by scenario
                    if scenario is not None:
                        colors.append(plt.cm.Set1(scenario % 10))
                    else:
                        colors.append('gray')

            if values:
                bars = plt.bar(range(len(values)), values, color=colors, alpha=0.7)
                plt.xticks(range(len(values)), labels, rotation=45, ha='center')
                plt.tick_params(axis='x', pad=1)
                plt.ylabel(title)
                plt.title(f'{title} Comparison')
                plt.grid(True, axis='y', alpha=0.3)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    plt.annotate(f'{value:.4f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'performance_comparison.png'),
                       bbox_inches='tight', dpi=150)
            print(f"✓ Saved performance comparison to {output_dir}/performance_comparison.png")
        plt.show()

    def compare_timing(self, save_plots=True, output_dir=None):
        """Compare timing metrics across runs."""
        if len(self.runs) < 2:
            print("Need at least 2 runs to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(18, 6))

        # Timing metrics comparison - 3 most important metrics in horizontal layout
        timing_metrics = ['avg_total_time', 'avg_resolution_time_ms', 'avg_aggregation_time']
        timing_titles = ['Avg Total Time (s)', 'Avg Resolution Time (ms)', 'Avg Aggregation Time (s)']

        for i, (metric, title) in enumerate(zip(timing_metrics, timing_titles)):
            plt.subplot(1, 3, i+1)

            values = []
            labels = []

            for run_name, run_data in self.runs.items():
                value = run_data.get(metric)
                if value is not None:
                    values.append(value)
                    scenario = run_data.get('scenario', 'Unknown')
                    clients = run_data.get('num_clients', 'Unknown')
                    labels.append(f"S{scenario}\n({clients} clients)")

            if values:
                # Use blue colors for all bars
                bars = plt.bar(range(len(values)), values, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1)
                plt.xticks(range(len(values)), labels, rotation=45, ha='center')
                plt.tick_params(axis='x', pad=1)
                plt.ylabel(title)
                plt.title(f'{title} Comparison')
                plt.grid(True, axis='y', alpha=0.3)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    if metric == 'avg_resolution_time_ms':
                        plt.annotate(f'{value:.3f}ms',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)
                    else:
                        plt.annotate(f'{value:.3f}s',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'timing_comparison.png'),
                       bbox_inches='tight', dpi=150)
            print(f"✓ Saved timing comparison to {output_dir}/timing_comparison.png")
        plt.show()

    def compare_round_progression(self, save_plots=True, output_dir=None):
        """Compare accuracy progression across rounds."""
        if len(self.runs) < 2:
            print("Need at least 2 runs to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))

        for run_name, run_data in self.runs.items():
            fl_results = run_data.get("fl_results", {})
            rounds_data = fl_results.get("rounds", [])

            if rounds_data:
                rounds = [r["round"] for r in rounds_data if r["round"] > 0]
                accuracies = [r["test_accuracy"] for r in rounds_data if r["round"] > 0]

                scenario = run_data.get('scenario', 'Unknown')
                clients = run_data.get('num_clients', 'Unknown')
                label = f"S{scenario} ({clients} clients)"

                plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=6, label=label)

        plt.xlabel('Round')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy Progression Across Rounds')
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_plots:
            plt.savefig(os.path.join(output_dir, 'accuracy_progression.png'),
                       bbox_inches='tight', dpi=150)
            print(f"✓ Saved accuracy progression to {output_dir}/accuracy_progression.png")
        plt.show()

    def compare_combined_metrics(self, save_plots=True, output_dir=None):
        """Create a combined plot with accuracy progression, resolution time, and aggregation time."""
        if len(self.runs) < 2:
            print("Need at least 2 runs to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(18, 6))

        # Subplot 1: Accuracy Progression
        plt.subplot(1, 3, 1)
        for run_name, run_data in self.runs.items():
            fl_results = run_data.get("fl_results", {})
            rounds_data = fl_results.get("rounds", [])

            if rounds_data:
                rounds = [r["round"] for r in rounds_data if r["round"] > 0]
                accuracies = [r["test_accuracy"] for r in rounds_data if r["round"] > 0]

                scenario = run_data.get('scenario', 'Unknown')
                clients = run_data.get('num_clients', 'Unknown')
                label = f"S{scenario} ({clients} clients)"

                plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=6, label=label)

        plt.xlabel('Round')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy Progression')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Subplot 2: Average Resolution Time (ms)
        plt.subplot(1, 3, 2)
        values = []
        labels = []

        for run_name, run_data in self.runs.items():
            value = run_data.get('avg_resolution_time_ms')
            if value is not None:
                values.append(value)
                scenario = run_data.get('scenario', 'Unknown')
                clients = run_data.get('num_clients', 'Unknown')
                labels.append(f"S{scenario}\n({clients} clients)")

        if values:
            bars = plt.bar(range(len(values)), values, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1)
            plt.xticks(range(len(values)), labels, rotation=45, ha='center')
            plt.tick_params(axis='x', pad=1)
            plt.ylabel('Avg Resolution Time (ms)')
            plt.title('Avg Resolution Time Comparison')
            plt.grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.annotate(f'{value:.3f}ms',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        # Subplot 3: Average Aggregation Time (s)
        plt.subplot(1, 3, 3)
        values = []
        labels = []

        for run_name, run_data in self.runs.items():
            value = run_data.get('avg_aggregation_time')
            if value is not None:
                values.append(value)
                scenario = run_data.get('scenario', 'Unknown')
                clients = run_data.get('num_clients', 'Unknown')
                labels.append(f"S{scenario}\n({clients} clients)")

        if values:
            bars = plt.bar(range(len(values)), values, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1)
            plt.xticks(range(len(values)), labels, rotation=45, ha='center')
            plt.tick_params(axis='x', pad=1)
            plt.ylabel('Avg Aggregation Time (s)')
            plt.title('Avg Aggregation Time Comparison')
            plt.grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.annotate(f'{value:.3f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'combined_metrics_comparison.png'),
                       bbox_inches='tight', dpi=150)
            print(f"✓ Saved combined metrics comparison to {output_dir}/combined_metrics_comparison.png")
        plt.show()

    def print_summary(self):
        """Print a summary comparison of all loaded runs."""
        if not self.runs:
            print("No runs loaded")
            return

        print("\n" + "="*80)
        print("FEDERATED LEARNING RUNS COMPARISON SUMMARY")
        print("="*80)

        for run_name, run_data in self.runs.items():
            print(f"\n📊 {run_name}")
            print(f"   Scenario: {run_data.get('scenario', 'Unknown')}")
            print(f"   Clients: {run_data.get('num_clients', 'Unknown')}")
            print(f"   Experiment: {run_data.get('experiment_type', 'Unknown')}")
            print(f"   Total Rounds: {run_data.get('total_rounds', 'Unknown')}")

            # Performance
            accuracy = run_data.get('final_accuracy')
            if accuracy:
                print(f"   Final Accuracy: {accuracy:.4f}")

            # Timing
            total_time = run_data.get('avg_total_time')
            resolution_time = run_data.get('avg_resolution_time_ms')
            overhead = run_data.get('disagreement_overhead_pct')

            if total_time:
                print(f"   Avg Total Time: {total_time:.3f}s")
            if resolution_time:
                print(f"   Avg Resolution Time: {resolution_time:.3f}ms")
            if overhead is not None:
                print(f"   Disagreement Overhead: {overhead:.1f}%")

            rounds_with_disag = run_data.get('rounds_with_disagreements', 0)
            total_timing_rounds = run_data.get('total_timing_rounds', 0)
            if total_timing_rounds > 0:
                print(f"   Rounds with Disagreements: {rounds_with_disag}/{total_timing_rounds}")


def main():
    parser = argparse.ArgumentParser(description='Compare Federated Learning Runs')
    parser.add_argument('runs', nargs='+', help='Paths to FL simulation result directories')

    # Generate timestamped default directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = f'results/comparisons/comparison_{timestamp}'

    parser.add_argument('--output-dir', '-o', default=default_output_dir,
                       help=f'Output directory for plots (default: {default_output_dir})')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots, only show summary')
    parser.add_argument('--names', nargs='+',
                       help='Custom names for the runs (optional)')

    args = parser.parse_args()

    # Initialize comparator
    comparator = FLRunComparator()

    # Load runs
    for i, run_path in enumerate(args.runs):
        custom_name = args.names[i] if args.names and i < len(args.names) else None
        comparator.load_run(run_path, custom_name)

    # Print summary
    comparator.print_summary()

    if not args.no_plots:
        print(f"\n📈 Generating comparison plots...")

        # Generate comparisons
        comparator.compare_performance(save_plots=True, output_dir=args.output_dir)
        comparator.compare_timing(save_plots=True, output_dir=args.output_dir)
        comparator.compare_round_progression(save_plots=True, output_dir=args.output_dir)
        comparator.compare_combined_metrics(save_plots=True, output_dir=args.output_dir)

        print(f"\n✅ All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
