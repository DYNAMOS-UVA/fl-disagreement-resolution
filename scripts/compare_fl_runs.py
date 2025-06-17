#!/usr/bin/env python3
"""
Federated Learning Run Comparison Tool

This script compares multiple federated learning simulation runs,
analyzing performance metrics, timing data, and scenario characteristics.
Can handle multiple runs per scenario and average the results.
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class FLRunComparator:
    def __init__(self):
        self.runs = {}
        self.scenario_runs = defaultdict(list)  # Group runs by scenario
        self.num_runs_per_scenario = {}  # Track number of runs per scenario

    def _get_scenario_description(self, scenario, clients):
        """Generate descriptive scenario labels based on scenario number."""
        if scenario is None or scenario == 'Unknown':
            return f"({clients} clients)"

        scenario_num = int(scenario) if isinstance(scenario, (int, str)) and str(scenario).isdigit() else scenario

        # Handle scenario 7 (10 clients, no exclusion)
        if scenario_num == 7:
            return f"({clients} clients,\nno excl.)"

        # Handle scenarios 8-12 (ring of 10)
        elif 8 <= scenario_num <= 12:
            return f"({clients} clients,\nring of 10)"

        # Handle scenarios 13-19 (6 clients with different exclusion patterns)
        elif 13 <= scenario_num <= 19:
            if scenario_num == 13:
                return "(no excl.)"
            else:
                # S14 = next 1, S15 = next 2, etc.
                next_count = scenario_num - 13
                return f"(next {next_count})"

        # Handle scenarios 20-24 (20 clients with ring patterns)
        elif 20 <= scenario_num <= 24:
            if scenario_num == 20:
                return "(no excl.)"
            elif scenario_num == 24:
                return f"({clients} clients,\nring of 20)"
            else:
                # S21 = ring of 5, S22 = ring of 10, S23 = ring of 15
                ring_size = (scenario_num - 20) * 5
                return f"(ring of {ring_size})"

        # Handle scenarios 25-28 (5 clients with ring patterns)
        elif 25 <= scenario_num <= 28:
            if scenario_num == 25:
                return f"({clients} clients,\nno excl.)"
            elif scenario_num == 26:
                return f"(ring of 5,\nnext 1)"
            else:
                # S27 = ring of 10, S28 = ring of 15
                ring_size = (scenario_num - 25) * 5
                return f"({clients} clients,\nring of {ring_size})"

        # Handle scenarios 29-31 (ring patterns with next exclusions)
        elif 29 <= scenario_num <= 31:
            if scenario_num == 29:
                return f"(ring of 10,\nnext 2)"
            elif scenario_num == 30:
                return f"(ring of 10,\nnext 3)"
            elif scenario_num == 31:
                return f"(ring of 10,\nnext 4)"

        # Fallback for other scenarios
        else:
            return f"({clients} clients)"

    def _generate_chart_label(self, scenario, clients):
        """Generate a label for charts (with newline)."""
        desc = self._get_scenario_description(scenario, clients)
        return f"S{scenario}\n{desc}"

    def _generate_legend_label(self, scenario, clients):
        """Generate a label for legends (without newline)."""
        desc = self._get_scenario_description(scenario, clients)
        return f"S{scenario} {desc}"

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

        # Group by scenario for averaging
        scenario = run_data.get('scenario')
        if scenario is not None:
            self.scenario_runs[scenario].append(run_data)

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

            # Calculate average track performance for each round
            experiment_type = run_data.get("experiment_type", "mnist")
            run_data["avg_track_performance"] = self._calculate_average_track_performance(rounds_data, experiment_type)

            # Calculate final average track performance metrics
            if run_data["avg_track_performance"]:
                final_round_num = max(run_data["avg_track_performance"].keys())
                run_data["final_avg_track_accuracy"] = run_data["avg_track_performance"][final_round_num]

                # Also calculate average track metrics from the final round's track results
                final_round = rounds_data[-1]
                track_results = final_round.get("track_results", {})

                if track_results:
                    # Collect all track metrics (including global)
                    track_precisions = []
                    track_recalls = []
                    track_f1s = []

                    # Add global metrics
                    if final_round.get("mean_precision") is not None:
                        track_precisions.append(final_round.get("mean_precision"))
                    if final_round.get("mean_recall") is not None:
                        track_recalls.append(final_round.get("mean_recall"))
                    if final_round.get("mean_f1") is not None:
                        track_f1s.append(final_round.get("mean_f1"))

                    # Add track-specific metrics
                    for track_name, track_data in track_results.items():
                        if track_data.get("precision") is not None:
                            track_precisions.append(track_data.get("precision"))
                        if track_data.get("recall") is not None:
                            track_recalls.append(track_data.get("recall"))
                        if track_data.get("f1") is not None:
                            track_f1s.append(track_data.get("f1"))

                    # Calculate averages
                    run_data["final_avg_track_precision"] = np.mean(track_precisions) if track_precisions else None
                    run_data["final_avg_track_recall"] = np.mean(track_recalls) if track_recalls else None
                    run_data["final_avg_track_f1"] = np.mean(track_f1s) if track_f1s else None

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

    def _calculate_average_track_performance(self, rounds_data, experiment_type=None):
        """Calculate average performance across all tracks (including global) for each round.

        Args:
            rounds_data: List of round data from fl_results
            experiment_type: Type of experiment ('mnist' or 'n_cmapss')

        Returns:
            dict: Dictionary with round numbers as keys and average performance as values
        """
        avg_performance = {}

        for round_data in rounds_data:
            if round_data["round"] <= 0:
                continue

            round_num = round_data["round"]
            track_results = round_data.get("track_results", {})

            # Collect all track performances (including global)
            performances = []

            # Handle different experiment types
            if experiment_type == "n_cmapss":
                # For N-CMAPSS, use test_loss (RMSE) instead of accuracy
                global_rmse = round_data.get("test_loss")
                if global_rmse is not None:
                    performances.append(global_rmse)

                # Add track-specific RMSE
                for track_name, track_data in track_results.items():
                    track_rmse = track_data.get("rmse")
                    if track_rmse is not None:
                        performances.append(track_rmse)

                # Calculate average if we have any performances
                if performances:
                    avg_performance[round_num] = np.mean(performances)
                else:
                    # Fallback to global performance if no track results
                    avg_performance[round_num] = global_rmse
            else:
                # Default to MNIST behavior (accuracy)
                global_accuracy = round_data.get("test_accuracy")
                if global_accuracy is not None:
                    performances.append(global_accuracy)

                # Add track-specific performances
                for track_name, track_data in track_results.items():
                    track_accuracy = track_data.get("accuracy")
                    if track_accuracy is not None:
                        performances.append(track_accuracy)

                # Calculate average if we have any performances
                if performances:
                    avg_performance[round_num] = np.mean(performances)
                else:
                    # Fallback to global performance if no track results
                    avg_performance[round_num] = global_accuracy

        return avg_performance

    def compare_performance(self, save_plots=True, output_dir=None):
        """Compare performance metrics across runs."""
        if len(self.scenario_runs) < 2:
            print("Need at least 2 scenarios to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        # Use averaged scenario data
        averaged_data = self.get_averaged_scenario_data()
        max_runs = max(self.num_runs_per_scenario.values()) if self.num_runs_per_scenario else 0

        # Performance comparison
        plt.figure(figsize=(15, 10))

        # Prepare data for comparison
        metrics = ['final_avg_track_accuracy', 'final_avg_track_precision', 'final_avg_track_recall', 'final_avg_track_f1']
        metric_titles = ['Final Avg Track Accuracy', 'Final Avg Track Precision', 'Final Avg Track Recall', 'Final Avg Track F1 Score']
        fallback_metrics = ['final_accuracy', 'final_precision', 'final_recall', 'final_f1']

        for i, (metric, title, fallback) in enumerate(zip(metrics, metric_titles, fallback_metrics)):
            plt.subplot(2, 2, i+1)

            values = []
            labels = []
            colors = []

            for scenario, run_data in averaged_data.items():
                # Try to get average track metric first, fallback to global metric
                value = run_data.get(metric)
                if value is None:
                    value = run_data.get(fallback)

                if value is not None:
                    values.append(value)
                    clients = run_data.get('num_clients', 'Unknown')
                    labels.append(self._generate_chart_label(scenario, clients))

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
                plt.title(f'{title} Comparison\n(across {max_runs} runs)')
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
        if len(self.scenario_runs) < 2:
            print("Need at least 2 scenarios to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        # Use averaged scenario data
        averaged_data = self.get_averaged_scenario_data()
        max_runs = max(self.num_runs_per_scenario.values()) if self.num_runs_per_scenario else 0

        plt.figure(figsize=(18, 6))

        # Timing metrics comparison - 3 most important metrics in horizontal layout
        timing_metrics = ['avg_total_time', 'avg_resolution_time_ms', 'avg_aggregation_time']
        timing_titles = ['Avg Total Time (s)', 'Avg Resolution Time (ms)', 'Avg Aggregation Time (s)']

        for i, (metric, title) in enumerate(zip(timing_metrics, timing_titles)):
            plt.subplot(1, 3, i+1)

            values = []
            labels = []

            for scenario, run_data in averaged_data.items():
                value = run_data.get(metric)
                if value is not None:
                    values.append(value)
                    clients = run_data.get('num_clients', 'Unknown')
                    labels.append(self._generate_chart_label(scenario, clients))

            if values:
                # Use blue colors for all bars
                bars = plt.bar(range(len(values)), values, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1)
                plt.xticks(range(len(values)), labels, rotation=45, ha='center')
                plt.tick_params(axis='x', pad=1)
                plt.ylabel(title)
                plt.title(f'{title} Comparison\n(across {max_runs} runs)')
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
        """Compare average track accuracy progression across rounds."""
        if len(self.scenario_runs) < 2:
            print("Need at least 2 scenarios to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        # Use averaged scenario data
        averaged_data = self.get_averaged_scenario_data()
        max_runs = max(self.num_runs_per_scenario.values()) if self.num_runs_per_scenario else 0

        # Detect experiment type from the first scenario
        first_scenario_data = next(iter(averaged_data.values())) if averaged_data else {}
        experiment_type = first_scenario_data.get("experiment_type", "mnist")

        plt.figure(figsize=(12, 8))

        for scenario, run_data in averaged_data.items():
            avg_track_performance = run_data.get("avg_track_performance", {})

            if avg_track_performance:
                rounds = list(avg_track_performance.keys())
                values = list(avg_track_performance.values())

                clients = run_data.get('num_clients', 'Unknown')
                label = self._generate_legend_label(scenario, clients)

                plt.plot(rounds, values, marker='o', linewidth=2, markersize=6, label=label)

        plt.xlabel('Round')

        if experiment_type == "n_cmapss":
            plt.ylabel('Average Track RMSE')
            plt.title(f'Average Track RMSE Progression Across Rounds\n(across {max_runs} runs)')
        else:
            plt.ylabel('Average Track Accuracy')
            plt.title(f'Average Track Accuracy Progression Across Rounds\n(across {max_runs} runs)')

        plt.grid(True, alpha=0.3)
        plt.legend()

        # Set x-axis to show only whole numbers
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if save_plots:
            plt.savefig(os.path.join(output_dir, 'accuracy_progression.png'),
                       bbox_inches='tight', dpi=150)
            print(f"✓ Saved accuracy progression to {output_dir}/accuracy_progression.png")
        plt.show()

    def compare_combined_metrics(self, save_plots=True, output_dir=None):
        """Create a combined plot with resolution time, aggregation time, and accuracy progression."""
        if len(self.scenario_runs) < 2:
            print("Need at least 2 scenarios to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        # Use averaged scenario data
        averaged_data = self.get_averaged_scenario_data()

        # Determine total number of runs across all scenarios for title
        total_runs = sum(self.num_runs_per_scenario.values())
        max_runs = max(self.num_runs_per_scenario.values()) if self.num_runs_per_scenario else 0

        plt.figure(figsize=(18, 6))

        # Subplot 1: Average Resolution Time (ms)
        plt.subplot(1, 3, 1)
        values = []
        labels = []

        for scenario, run_data in averaged_data.items():
            value = run_data.get('avg_resolution_time_ms')
            if value is not None:
                values.append(value)
                clients = run_data.get('num_clients', 'Unknown')
                labels.append(self._generate_chart_label(scenario, clients))

        if values:
            bars = plt.bar(range(len(values)), values, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1)
            plt.xticks(range(len(values)), labels, rotation=45, ha='center')
            plt.tick_params(axis='x', pad=1)
            plt.ylabel('Average Resolution Time (ms)')
            plt.title(f'Average Resolution Time Comparison\n(across {max_runs} runs)')
            plt.grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.annotate(f'{value:.3f}ms',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        # Subplot 2: Average Aggregation Time (s)
        plt.subplot(1, 3, 2)
        values = []
        labels = []

        for scenario, run_data in averaged_data.items():
            value = run_data.get('avg_aggregation_time')
            if value is not None:
                values.append(value)
                clients = run_data.get('num_clients', 'Unknown')
                labels.append(self._generate_chart_label(scenario, clients))

        if values:
            bars = plt.bar(range(len(values)), values, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1)
            plt.xticks(range(len(values)), labels, rotation=45, ha='center')
            plt.tick_params(axis='x', pad=1)
            plt.ylabel('Average Aggregation Time (s)')
            plt.title(f'Average Aggregation Time Comparison\n(across {max_runs} runs)')
            plt.grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.annotate(f'{value:.3f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        # Subplot 3: Accuracy Progression or RMSE Progression
        plt.subplot(1, 3, 3)

        # Detect experiment type from the first scenario
        first_scenario_data = next(iter(averaged_data.values())) if averaged_data else {}
        experiment_type = first_scenario_data.get("experiment_type", "mnist")

        for scenario, run_data in averaged_data.items():
            avg_track_performance = run_data.get("avg_track_performance", {})

            if avg_track_performance:
                rounds = list(avg_track_performance.keys())
                values = list(avg_track_performance.values())

                clients = run_data.get('num_clients', 'Unknown')
                label = self._generate_legend_label(scenario, clients)

                plt.plot(rounds, values, marker='o', linewidth=2, markersize=6, label=label)

        plt.xlabel('Round')

        if experiment_type == "n_cmapss":
            plt.ylabel('Average Track RMSE')
            plt.title(f'Average Track RMSE Progression\n(across {max_runs} runs)')
        else:
            plt.ylabel('Average Track Accuracy')
            plt.title(f'Average Track Accuracy Progression\n(across {max_runs} runs)')

        plt.grid(True, alpha=0.3)
        plt.legend()

        # Set x-axis to show only whole numbers
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

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
        print("FEDERATED LEARNING RUNS COMPARISON SUMMARY (AVERAGED BY SCENARIO)")
        print("="*80)

        # Print individual runs first
        print("\n📝 Individual runs loaded:")
        for run_name, run_data in self.runs.items():
            scenario = run_data.get('scenario', 'Unknown')
            print(f"   • {run_name} (Scenario {scenario})")

        # Then print averaged summary
        averaged_data = self.get_averaged_scenario_data()

        for scenario, run_data in averaged_data.items():
            num_runs = self.num_runs_per_scenario.get(scenario, 0)
            print(f"\n📊 Scenario {scenario} (averaged across {num_runs} runs)")
            print(f"   Clients: {run_data.get('num_clients', 'Unknown')}")
            print(f"   Experiment: {run_data.get('experiment_type', 'Unknown')}")
            print(f"   Total Rounds: {run_data.get('total_rounds', 'Unknown'):.1f}")

            # Performance
            accuracy = run_data.get('final_accuracy')
            avg_track_accuracy = run_data.get('final_avg_track_accuracy')

            if accuracy:
                print(f"   Final Global Accuracy: {accuracy:.4f}")
            if avg_track_accuracy and avg_track_accuracy != accuracy:
                print(f"   Final Avg Track Accuracy: {avg_track_accuracy:.4f}")
            elif avg_track_accuracy:
                print(f"   Final Accuracy: {avg_track_accuracy:.4f}")

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
                print(f"   Rounds with Disagreements: {rounds_with_disag:.1f}/{total_timing_rounds:.1f}")

    def _average_scenario_metrics(self, scenario_runs):
        """Average metrics across multiple runs of the same scenario."""
        if not scenario_runs:
            return None

        # Initialize aggregated data with the first run's structure
        first_run = scenario_runs[0]
        avg_data = {
            "scenario": first_run.get("scenario"),
            "num_clients": first_run.get("num_clients"),
            "experiment_type": first_run.get("experiment_type"),
            "num_runs": len(scenario_runs)
        }

        # Metrics to average (numeric values)
        numeric_metrics = [
            'final_accuracy', 'final_loss', 'final_precision', 'final_recall', 'final_f1',
            'final_avg_track_accuracy', 'final_avg_track_precision', 'final_avg_track_recall', 'final_avg_track_f1',
            'avg_total_time', 'avg_aggregation_time', 'avg_resolution_time_ms',
            'disagreement_overhead_pct', 'total_rounds'
        ]

        # Average numeric metrics
        for metric in numeric_metrics:
            values = [run.get(metric) for run in scenario_runs if run.get(metric) is not None]
            if values:
                avg_data[metric] = np.mean(values)

        # Sum integer metrics
        integer_sum_metrics = ['rounds_with_disagreements', 'total_timing_rounds']
        for metric in integer_sum_metrics:
            values = [run.get(metric, 0) for run in scenario_runs if run.get(metric) is not None]
            if values:
                avg_data[metric] = int(np.mean(values))

        # Average track performance across rounds
        experiment_type = first_run.get("experiment_type", "mnist")
        avg_data["avg_track_performance"] = self._average_track_performance_across_runs(scenario_runs, experiment_type)

        return avg_data

    def _average_track_performance_across_runs(self, scenario_runs, experiment_type="mnist"):
        """Average track performance across multiple runs for each round."""
        if not scenario_runs:
            return {}

        # Collect all rounds from all runs
        all_rounds = set()
        for run in scenario_runs:
            track_perf = run.get("avg_track_performance", {})
            all_rounds.update(track_perf.keys())

        if not all_rounds:
            return {}

        # Average performance for each round
        avg_track_performance = {}
        for round_num in sorted(all_rounds):
            round_values = []
            for run in scenario_runs:
                track_perf = run.get("avg_track_performance", {})
                if round_num in track_perf:
                    round_values.append(track_perf[round_num])

            if round_values:
                avg_track_performance[round_num] = np.mean(round_values)

        return avg_track_performance

    def get_averaged_scenario_data(self):
        """Get averaged data for each scenario."""
        averaged_data = {}

        for scenario, runs in self.scenario_runs.items():
            self.num_runs_per_scenario[scenario] = len(runs)
            averaged_data[scenario] = self._average_scenario_metrics(runs)

        return averaged_data

def main():
    parser = argparse.ArgumentParser(
        description='Compare Federated Learning Runs - Automatically averages multiple runs of the same scenario',
        epilog='The script automatically groups runs by scenario and averages their metrics. '
               'If you have multiple runs of the same scenario, they will be averaged together in the comparison charts.')
    parser.add_argument('runs', nargs='+', help='Paths to FL simulation result directories (multiple runs of the same scenario will be averaged)')

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
        print(f"\n📈 Generating comparison plots (averaged by scenario)...")

        # Generate comparisons
        comparator.compare_performance(save_plots=True, output_dir=args.output_dir)
        comparator.compare_timing(save_plots=True, output_dir=args.output_dir)
        comparator.compare_round_progression(save_plots=True, output_dir=args.output_dir)
        comparator.compare_combined_metrics(save_plots=True, output_dir=args.output_dir)

        print(f"\n✅ All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
