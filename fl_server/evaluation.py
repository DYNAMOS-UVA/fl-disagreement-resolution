"""Model evaluation functionality for federated learning."""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import copy

from fl_server.utils import make_json_serializable, read_client_results_from_files

def evaluate_model(server, fl_round=None, client_results=None):
    """Evaluate global model on test data.

    Args:
        server: FederatedServer instance
        fl_round: Current federated learning round (if None, considered as initial round 0)
        client_results: Dictionary of client training results (optional)

    Returns:
        tuple: (test_loss, accuracy) where accuracy is None for regression tasks
    """
    server.global_model.eval()

    # If round is not provided, use the current round counter
    if fl_round is None:
        fl_round = server.round
    else:
        # Update the internal round counter if a specific round is provided
        server.round = fl_round

    # Read client results from filesystem if not round 0
    if fl_round > 0 and not client_results and server.results_dir and server.client_ids:
        client_results = read_client_results_from_files(server.results_dir, server.client_ids, fl_round)

    # Set criterion based on experiment type
    if server.experiment_type == "n_cmapss":
        criterion = nn.MSELoss()
    elif server.experiment_type == "mnist":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown experiment type: {server.experiment_type}")

    test_loss = 0
    predictions = []
    actual = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in server.test_loader:
            data, target = data.to(server.device), target.to(server.device)
            output = server.global_model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            # For regression (N-CMAPSS)
            if server.experiment_type == "n_cmapss":
                predictions.extend(output.cpu().numpy())
                actual.extend(target.cpu().numpy())
            # For classification (MNIST)
            elif server.experiment_type == "mnist":
                _, predicted = torch.max(output.data, 1)
                predictions.extend(predicted.cpu().numpy())
                actual.extend(target.cpu().numpy())
                total += target.size(0)
                correct += (predicted == target).sum().item()

    # Calculate average test loss
    test_loss /= len(server.test_loader)

    # For RUL prediction, calculate RMSE and additional metrics
    if server.experiment_type == "n_cmapss":
        rmse = np.sqrt(test_loss)
        test_loss = rmse  # Keep RMSE as the primary test loss metric

        # Convert to numpy arrays for calculation
        predictions = np.array(predictions)
        actual = np.array(actual)

        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(predictions - actual))

        # Calculate R² (coefficient of determination)
        mean_actual = np.mean(actual)
        ss_total = np.sum((actual - mean_actual) ** 2)
        ss_residual = np.sum((actual - predictions) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        # Calculate % of predictions within ±10 cycles (a more intuitive metric)
        within_10_cycles = np.mean(np.abs(predictions - actual) <= 10.0) * 100
        within_20_cycles = np.mean(np.abs(predictions - actual) <= 20.0) * 100

        # Print all metrics
        print(f"Round {server.round} - RUL Prediction Metrics:")
        print(f"  RMSE: {rmse:.2f} cycles")
        print(f"  MAE: {mae:.2f} cycles")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Within ±10 cycles: {within_10_cycles:.2f}%")
        print(f"  Within ±20 cycles: {within_20_cycles:.2f}%")

        # Store additional metrics in training history
        if "rul_mae" not in server.training_history:
            server.training_history["rul_mae"] = []
        if "rul_r_squared" not in server.training_history:
            server.training_history["rul_r_squared"] = []
        if "rul_within_10" not in server.training_history:
            server.training_history["rul_within_10"] = []
        if "rul_within_20" not in server.training_history:
            server.training_history["rul_within_20"] = []

        server.training_history["rul_mae"].append(mae)
        server.training_history["rul_r_squared"].append(r_squared)
        server.training_history["rul_within_10"].append(within_10_cycles)
        server.training_history["rul_within_20"].append(within_20_cycles)

        # Update results dictionary
        round_results = {
            "round": fl_round,
            "test_loss": test_loss,
            "mae": mae,
            "r_squared": r_squared,
            "within_10_cycles": within_10_cycles,
            "within_20_cycles": within_20_cycles
        }

        # Add client results if provided
        if client_results:
            round_results["client_results"] = client_results

        # Add to results history
        server.results["rounds"].append(round_results)

        accuracy = None

    # For classification, calculate accuracy and other metrics
    elif server.experiment_type == "mnist":
        accuracy = correct / total

        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predictions, average=None, zero_division=0
        )

        # Calculate mean metrics (weighted by support)
        mean_precision, mean_recall, mean_f1, _ = precision_recall_fscore_support(
            actual, predictions, average='weighted', zero_division=0
        )

        # Calculate per-class accuracy
        class_labels = np.unique(actual)
        per_class_accuracy = []
        for c in class_labels:
            # Mask for this class
            mask = np.array(actual) == c
            # Accuracy for this class
            class_acc = np.mean(np.array(predictions)[mask] == c) if np.sum(mask) > 0 else 0
            per_class_accuracy.append(class_acc)

        # Print detailed metrics
        print(f"Round {server.round} - MNIST Classification Metrics:")
        print(f"  Overall Accuracy: {accuracy:.4f}")
        print(f"  Mean Precision: {mean_precision:.4f}")
        print(f"  Mean Recall: {mean_recall:.4f}")
        print(f"  Mean F1 Score: {mean_f1:.4f}")

        # Store additional metrics in training history
        if "mnist_precision" not in server.training_history:
            server.training_history["mnist_precision"] = []
        if "mnist_recall" not in server.training_history:
            server.training_history["mnist_recall"] = []
        if "mnist_f1" not in server.training_history:
            server.training_history["mnist_f1"] = []
        if "mnist_per_class_accuracy" not in server.training_history:
            server.training_history["mnist_per_class_accuracy"] = []

        server.training_history["mnist_precision"].append(mean_precision)
        server.training_history["mnist_recall"].append(mean_recall)
        server.training_history["mnist_f1"].append(mean_f1)
        server.training_history["mnist_per_class_accuracy"].append(per_class_accuracy)

        print(f"Round {server.round} - Global model test accuracy: {accuracy:.4f}")

        # Update results dictionary
        round_results = {
            "round": fl_round,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "mean_precision": mean_precision if 'mean_precision' in locals() else None,
            "mean_recall": mean_recall if 'mean_recall' in locals() else None,
            "mean_f1": mean_f1 if 'mean_f1' in locals() else None
        }

        # Add client results if provided
        if client_results:
            round_results["client_results"] = client_results

        # Add to results history
        server.results["rounds"].append(round_results)

    # Store history
    server.training_history["rounds"].append(server.round)
    server.training_history["global_test_loss"].append(test_loss)
    if accuracy is not None:
        server.training_history["global_test_accuracy"].append(accuracy)

    print(f"Round {server.round} - Global model test loss: {test_loss:.6f}")

    # Plot and save results
    save_evaluation_results(server, predictions, actual)

    # Evaluate each track model if this isn't round 0
    if fl_round > 0 and server.results_dir:
        print(f"\n=== EVALUATING TRACK MODELS FOR ROUND {fl_round} ===")
        track_results = evaluate_track_models(server, fl_round)

        # Store track results in the main results dictionary
        if track_results:
            # Add track results to the round results
            for round_result in server.results["rounds"]:
                if round_result["round"] == fl_round:
                    round_result["track_results"] = track_results
                    break

            # If no track results storage found in training history, create it
            if "track_results" not in server.training_history:
                server.training_history["track_results"] = {}

            # Store track results in training history
            server.training_history["track_results"][str(fl_round)] = track_results

            # Print summary of track results
            print(f"\n=== TRACK EVALUATION SUMMARY FOR ROUND {fl_round} ===")
            if accuracy is not None:
                print(f"Global model - Accuracy: {accuracy:.6f}")
            else:
                print(f"Global model - RMSE: {test_loss:.6f}")

            for track_name, track_data in track_results.items():
                if server.experiment_type == "mnist":
                    print(f"{track_name} - Accuracy: {track_data['accuracy']:.6f}")
                else:
                    print(f"{track_name} - RMSE: {track_data['rmse']:.6f}")

            print("=== END TRACK EVALUATION ===\n")

    # Save experiment results
    server._save_experiment_results()

    return test_loss, accuracy

def save_evaluation_results(server, predictions, actual):
    """Save evaluation results and generate plots.

    Args:
        server: FederatedServer instance
        predictions: List of model predictions
        actual: List of actual values
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine output paths based on results_dir
    if server.results_dir:
        history_path = os.path.join(server.output_dir, f"training_history_round_{server.round}.json")
        loss_plot_path = os.path.join(server.output_dir, "plots", f"global_model_loss_round_{server.round}.png")

        if server.experiment_type == "n_cmapss":
            pred_plot_path = os.path.join(server.output_dir, "plots", f"rul_prediction_round_{server.round}.png")
            metric_plot_path = os.path.join(server.output_dir, "plots", f"rul_metrics_round_{server.round}.png")
        else:
            cm_plot_path = os.path.join(server.output_dir, "plots", f"mnist_confusion_matrix_round_{server.round}.png")
            acc_plot_path = os.path.join(server.output_dir, "plots", f"global_model_accuracy_round_{server.round}.png")
    else:
        history_path = f"output/server_results/training_history_round_{server.round}_{timestamp}.json"
        loss_plot_path = f"output/plots/global_model_loss_round_{server.round}_{timestamp}.png"

        if server.experiment_type == "n_cmapss":
            pred_plot_path = f"output/plots/rul_prediction_round_{server.round}_{timestamp}.png"
            metric_plot_path = f"output/plots/rul_metrics_round_{server.round}_{timestamp}.png"
        else:
            cm_plot_path = f"output/plots/mnist_confusion_matrix_round_{server.round}_{timestamp}.png"
            acc_plot_path = f"output/plots/global_model_accuracy_round_{server.round}_{timestamp}.png"

    # Convert numpy values to Python native types for JSON serialization
    history_for_json = make_json_serializable(server.training_history)

    # Save training history
    with open(history_path, "w") as f:
        json.dump(history_for_json, f)

    # Plot and save loss history
    plt.figure(figsize=(10, 6))
    plt.plot(server.training_history["rounds"], server.training_history["global_test_loss"], marker='o')
    plt.xlabel('Federated Learning Round')
    plt.ylabel('Test Loss')
    plt.title(f'Global Model Performance ({server.experiment_type})')
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()

    # For RUL prediction, plot predictions vs actual
    if server.experiment_type == "n_cmapss":
        plot_ncmapss_results(server, predictions, actual, pred_plot_path, metric_plot_path)
    # For MNIST, plot confusion matrix and accuracy
    elif server.experiment_type == "mnist":
        plot_mnist_results(server, predictions, actual, cm_plot_path, acc_plot_path, timestamp)

    # Plot track progress if we have more than one round
    if server.round > 1 and "track_results" in server.training_history:
        plot_track_progress(server, server.round)

    # We don't need to save the model here as it's already saved in the round-specific directories
    # When the server calls save_model() during aggregation

    print(f"Saved results for round {server.round}")

def plot_ncmapss_results(server, predictions, actual, pred_plot_path, metric_plot_path):
    """Generate and save plots for N-CMAPSS results.

    Args:
        server: FederatedServer instance
        predictions: Numpy array of predictions
        actual: Numpy array of actual values
        pred_plot_path: Path to save prediction plot
        metric_plot_path: Path to save metrics plot
    """
    predictions = np.array(predictions)
    actual = np.array(actual)

    # Calculate error thresholds for coloring
    errors = predictions - actual
    within_10 = np.abs(errors) <= 10
    within_20 = np.logical_and(np.abs(errors) > 10, np.abs(errors) <= 20)
    beyond_20 = np.abs(errors) > 20

    # Create prediction scatter plot with colored points based on error
    plt.figure(figsize=(10, 6))

    # Plot points outside 20 cycles first (red)
    plt.scatter(actual[beyond_20], predictions[beyond_20], color='red', alpha=0.5, label='Error > 20 cycles')

    # Plot points within 10-20 cycles (yellow)
    plt.scatter(actual[within_20], predictions[within_20], color='orange', alpha=0.5, label='Error 10-20 cycles')

    # Plot points within 10 cycles last (green)
    plt.scatter(actual[within_10], predictions[within_10], color='green', alpha=0.5, label='Error ≤ 10 cycles')

    # Add perfect prediction line
    min_val = min(np.min(actual), np.min(predictions))
    max_val = max(np.max(actual), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

    # Add ±10 cycle lines
    plt.plot([min_val, max_val], [min_val + 10, max_val + 10], 'g--', alpha=0.3)
    plt.plot([min_val, max_val], [min_val - 10, max_val - 10], 'g--', alpha=0.3)

    # Add ±20 cycle lines
    plt.plot([min_val, max_val], [min_val + 20, max_val + 20], 'orange', linestyle='--', alpha=0.3)
    plt.plot([min_val, max_val], [min_val - 20, max_val - 20], 'orange', linestyle='--', alpha=0.3)

    plt.xlabel('Actual RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')

    # Get current metrics
    rmse = server.training_history["global_test_loss"][-1]
    mae = server.training_history["rul_mae"][-1]
    r2 = server.training_history["rul_r_squared"][-1]
    within_10_pct = server.training_history["rul_within_10"][-1]
    within_20_pct = server.training_history["rul_within_20"][-1]

    plt.title(f'RUL Prediction - Round {server.round}\n'
             f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}\n'
             f'Within ±10 cycles: {within_10_pct:.2f}%, Within ±20 cycles: {within_20_pct:.2f}%')

    plt.legend()
    plt.grid(True)
    plt.savefig(pred_plot_path)
    plt.close()

    # Plot additional metrics across rounds if we have at least 2 rounds
    if len(server.training_history["rounds"]) >= 2:
        plt.figure(figsize=(15, 10))

        # Create a 2x2 grid of subplots
        plt.subplot(2, 2, 1)
        plt.plot(server.training_history["rounds"], server.training_history["global_test_loss"], marker='o')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('RMSE (cycles)')
        plt.title('Root Mean Squared Error')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(server.training_history["rounds"], server.training_history["rul_mae"], marker='o', color='orange')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('MAE (cycles)')
        plt.title('Mean Absolute Error')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(server.training_history["rounds"], server.training_history["rul_within_10"], marker='o', color='green')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Percentage (%)')
        plt.title('Predictions Within ±10 Cycles')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(server.training_history["rounds"], server.training_history["rul_r_squared"], marker='o', color='purple')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('R²')
        plt.title('Coefficient of Determination (R²)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(metric_plot_path)
        plt.close()

def plot_mnist_results(server, predictions, actual, cm_plot_path, acc_plot_path, timestamp):
    """Generate and save plots for MNIST results.

    Args:
        server: FederatedServer instance
        predictions: Numpy array of predictions
        actual: Numpy array of actual values
        cm_plot_path: Path to save confusion matrix plot
        acc_plot_path: Path to save accuracy plot
        timestamp: Timestamp string for naming files
    """
    predictions = np.array(predictions)
    actual = np.array(actual)

    # Get the current metrics from the training history
    current_accuracy = server.training_history["global_test_accuracy"][-1]
    current_precision = server.training_history["mnist_precision"][-1]
    current_recall = server.training_history["mnist_recall"][-1]
    current_f1 = server.training_history["mnist_f1"][-1]
    current_per_class_acc = server.training_history["mnist_per_class_accuracy"][-1]

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(actual, predictions)
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a figure with two subplots for raw and normalized confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted Labels')
    ax1.set_ylabel('True Labels')
    ax1.set_title('Confusion Matrix (Raw Counts)')

    # Normalized by row (true label)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted Labels')
    ax2.set_ylabel('True Labels')
    ax2.set_title('Confusion Matrix (Normalized by True Label)')

    plt.suptitle(f'MNIST Classification Results - Round {server.round}')
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()

    # Plot metrics history if we have at least 2 rounds
    if len(server.training_history["rounds"]) >= 2:
        # Create a 2x2 subplot for accuracy, precision, recall, and F1 score
        plt.figure(figsize=(15, 10))

        # Accuracy
        plt.subplot(2, 2, 1)
        plt.plot(server.training_history["rounds"], server.training_history["global_test_accuracy"],
                 marker='o', color='blue')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Accuracy')
        plt.title('Overall Accuracy')
        plt.grid(True)

        # Precision
        plt.subplot(2, 2, 2)
        plt.plot(server.training_history["rounds"], server.training_history["mnist_precision"],
                 marker='o', color='green')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Precision')
        plt.title('Weighted Precision')
        plt.grid(True)

        # Recall
        plt.subplot(2, 2, 3)
        plt.plot(server.training_history["rounds"], server.training_history["mnist_recall"],
                 marker='o', color='orange')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Recall')
        plt.title('Weighted Recall')
        plt.grid(True)

        # F1 Score
        plt.subplot(2, 2, 4)
        plt.plot(server.training_history["rounds"], server.training_history["mnist_f1"],
                 marker='o', color='purple')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('F1 Score')
        plt.title('Weighted F1 Score')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(acc_plot_path)
        plt.close()

        # Plot per-class metrics for the current round
        num_classes = len(current_per_class_acc)
        plt.figure(figsize=(12, 6))

        # Retrieve per-class metrics for the latest round
        # We re-calculate precision, recall, and F1 scores per class
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            actual, predictions, average=None, zero_division=0
        )

        classes = np.arange(num_classes)
        x = np.arange(len(classes))
        width = 0.2

        # Bar chart with per-class metrics
        plt.bar(x - 1.5*width, current_per_class_acc, width, label='Accuracy', color='blue')
        plt.bar(x - 0.5*width, precision_per_class, width, label='Precision', color='green')
        plt.bar(x + 0.5*width, recall_per_class, width, label='Recall', color='orange')
        plt.bar(x + 1.5*width, f1_per_class, width, label='F1 Score', color='purple')

        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title(f'Per-class Metrics - Round {server.round}')
        plt.xticks(x, classes)
        plt.legend()
        plt.grid(True, axis='y')

        # Add the current round's metrics as a subtitle
        plt.figtext(0.5, 0.01,
                   f"Overall: Acc={current_accuracy:.4f}, Prec={current_precision:.4f}, Rec={current_recall:.4f}, F1={current_f1:.4f}",
                   ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

        if server.results_dir:
            per_class_path = os.path.join(server.output_dir, "plots", f"mnist_per_class_metrics_round_{server.round}.png")
        else:
            per_class_path = f"output/plots/mnist_per_class_metrics_round_{server.round}_{timestamp}.png"

        plt.tight_layout()
        plt.savefig(per_class_path)
        plt.close()

def evaluate_track_models(server, round_num):
    """Evaluate models for each track in a specific round.

    Args:
        server: FederatedServer instance
        round_num: Current federated learning round

    Returns:
        dict: Dictionary with track evaluation results
    """
    # Get structure configuration
    structure = server._get_structure_config()

    # Path to tracks directory
    tracks_dir = os.path.join(
        server.results_dir,
        structure["round_template"].format(round=round_num),
        "tracks"
    )

    # Check if tracks directory exists
    if not os.path.exists(tracks_dir):
        print(f"No tracks directory found for round {round_num}")

        # Check if there were tracks in previous rounds
        had_previous_tracks = False
        for prev_round in range(1, round_num):
            prev_tracks_dir = os.path.join(
                server.results_dir,
                structure["round_template"].format(round=prev_round),
                "tracks"
            )
            if os.path.exists(prev_tracks_dir):
                had_previous_tracks = True
                break

        # If there were tracks before but not now, it means disagreements have expired
        # Evaluate just the global model for comparison
        if had_previous_tracks:
            print(f"Disagreements have expired in round {round_num}, evaluating only the global model")

            # Set criterion based on experiment type
            if server.experiment_type == "n_cmapss":
                criterion = nn.MSELoss()
            elif server.experiment_type == "mnist":
                criterion = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown experiment type: {server.experiment_type}")

            # Path to global model for this round
            global_model_dir = os.path.join(
                server.results_dir,
                structure["round_template"].format(round=round_num),
                structure["global_model_aggregated"]
            )
            global_model_path = os.path.join(global_model_dir, "model.pt")

            if not os.path.exists(global_model_path):
                print(f"Global model file not found for round {round_num}")
                return {}

            # Save the current global model state
            original_state = copy.deepcopy(server.global_model.state_dict())

            # Load the global model
            server.global_model.load_state_dict(torch.load(global_model_path, map_location=server.device))
            server.global_model.eval()

            # Initialize metrics
            test_loss = 0
            predictions = []
            actual = []
            correct = 0
            total = 0

            # Evaluate the model
            with torch.no_grad():
                for data, target in server.test_loader:
                    data, target = data.to(server.device), target.to(server.device)
                    output = server.global_model(data)
                    loss = criterion(output, target)
                    test_loss += loss.item()

                    # For regression (N-CMAPSS)
                    if server.experiment_type == "n_cmapss":
                        predictions.extend(output.cpu().numpy())
                        actual.extend(target.cpu().numpy())
                    # For classification (MNIST)
                    elif server.experiment_type == "mnist":
                        _, predicted = torch.max(output.data, 1)
                        predictions.extend(predicted.cpu().numpy())
                        actual.extend(target.cpu().numpy())
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

            # Calculate average test loss
            test_loss /= len(server.test_loader)

            # Create results for global model
            track_results = {}

            if server.experiment_type == "n_cmapss":
                rmse = np.sqrt(test_loss)
                predictions = np.array(predictions)
                actual = np.array(actual)
                mae = np.mean(np.abs(predictions - actual))
                mean_actual = np.mean(actual)
                ss_total = np.sum((actual - mean_actual) ** 2)
                ss_residual = np.sum((actual - predictions) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                within_10_cycles = np.mean(np.abs(predictions - actual) <= 10.0) * 100
                within_20_cycles = np.mean(np.abs(predictions - actual) <= 20.0) * 100

                track_results["global"] = {
                    "rmse": rmse,
                    "mae": mae,
                    "r_squared": r_squared,
                    "within_10_cycles": within_10_cycles,
                    "within_20_cycles": within_20_cycles
                }

                print(f"Global model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r_squared:.4f}")

            elif server.experiment_type == "mnist":
                accuracy = correct / total if total > 0 else 0
                precision, recall, f1, _ = precision_recall_fscore_support(
                    actual, predictions, average='weighted', zero_division=0
                )

                track_results["global"] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "test_loss": test_loss
                }

                print(f"Global model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            # Restore the original global model state
            server.global_model.load_state_dict(original_state)

            # Save track results to a file
            if track_results and server.results_dir:
                results_path = os.path.join(
                    server.output_dir,
                    f"track_evaluation_round_{round_num}.json"
                )

                with open(results_path, 'w') as f:
                    json.dump(make_json_serializable(track_results), f, indent=2)

            return track_results

        return {}

    # Get track metadata
    metadata_path = os.path.join(tracks_dir, "track_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"No track metadata found for round {round_num}")
        return {}

    try:
        with open(metadata_path, 'r') as f:
            track_metadata = json.load(f)

        track_names = list(track_metadata.get("tracks", {}).keys())
        print(f"Found {len(track_names)} tracks to evaluate: {track_names}")

        # Initialize results
        track_results = {}

        # Set criterion based on experiment type
        if server.experiment_type == "n_cmapss":
            criterion = nn.MSELoss()
        elif server.experiment_type == "mnist":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown experiment type: {server.experiment_type}")

        # Save the current global model state
        original_state = copy.deepcopy(server.global_model.state_dict())

        # Evaluate each track model
        for track_name in track_names:
            track_dir = os.path.join(tracks_dir, track_name)
            model_path = os.path.join(track_dir, "model.pt")

            if not os.path.exists(model_path):
                print(f"Model file not found for track {track_name}")
                continue

            print(f"Evaluating track: {track_name}")

            # Load this track's model
            server.global_model.load_state_dict(torch.load(model_path, map_location=server.device))
            server.global_model.eval()

            # Initialize metrics
            test_loss = 0
            predictions = []
            actual = []
            correct = 0
            total = 0

            # Evaluate the model
            with torch.no_grad():
                for data, target in server.test_loader:
                    data, target = data.to(server.device), target.to(server.device)
                    output = server.global_model(data)
                    loss = criterion(output, target)
                    test_loss += loss.item()

                    # For regression (N-CMAPSS)
                    if server.experiment_type == "n_cmapss":
                        predictions.extend(output.cpu().numpy())
                        actual.extend(target.cpu().numpy())
                    # For classification (MNIST)
                    elif server.experiment_type == "mnist":
                        _, predicted = torch.max(output.data, 1)
                        predictions.extend(predicted.cpu().numpy())
                        actual.extend(target.cpu().numpy())
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

            # Calculate average test loss
            test_loss /= len(server.test_loader)

            # Get detailed metrics based on experiment type
            if server.experiment_type == "n_cmapss":
                rmse = np.sqrt(test_loss)

                # Convert to numpy arrays for calculation
                predictions = np.array(predictions)
                actual = np.array(actual)

                # Calculate Mean Absolute Error (MAE)
                mae = np.mean(np.abs(predictions - actual))

                # Calculate R² (coefficient of determination)
                mean_actual = np.mean(actual)
                ss_total = np.sum((actual - mean_actual) ** 2)
                ss_residual = np.sum((actual - predictions) ** 2)
                r_squared = 1 - (ss_residual / ss_total)

                # Calculate % of predictions within ±10 cycles (a more intuitive metric)
                within_10_cycles = np.mean(np.abs(predictions - actual) <= 10.0) * 100
                within_20_cycles = np.mean(np.abs(predictions - actual) <= 20.0) * 100

                track_results[track_name] = {
                    "rmse": rmse,
                    "mae": mae,
                    "r_squared": r_squared,
                    "within_10_cycles": within_10_cycles,
                    "within_20_cycles": within_20_cycles
                }

                print(f"Track '{track_name}' - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r_squared:.4f}")

            elif server.experiment_type == "mnist":
                accuracy = correct / total if total > 0 else 0

                # Calculate precision, recall, and F1 score
                precision, recall, f1, _ = precision_recall_fscore_support(
                    actual, predictions, average='weighted', zero_division=0
                )

                track_results[track_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "test_loss": test_loss
                }

                print(f"Track '{track_name}' - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Restore the original global model state
        server.global_model.load_state_dict(original_state)

        # Save track results to a file
        if track_results and server.results_dir:
            results_path = os.path.join(
                server.output_dir,
                f"track_evaluation_round_{round_num}.json"
            )

            with open(results_path, 'w') as f:
                json.dump(make_json_serializable(track_results), f, indent=2)

            # Plot track comparisons
            plot_track_comparison(server, track_results, round_num)

        return track_results

    except Exception as e:
        print(f"Error evaluating track models: {e}")
        import traceback
        traceback.print_exc()
        return {}

def plot_track_comparison(server, track_results, round_num):
    """Plot comparison of track performance metrics.

    Args:
        server: FederatedServer instance
        track_results: Dictionary with track evaluation results
        round_num: Current round number
    """
    if not track_results:
        return

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(server.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Check if this is a special case with only global track (expired disagreements)
    is_global_only = len(track_results) == 1 and "global" in track_results

    # Load previous round results to compare if disagreements have expired
    prev_results = None
    if is_global_only and round_num > 1:
        # Try to load previous round results
        prev_results_path = os.path.join(
            server.output_dir,
            f"track_evaluation_round_{round_num-1}.json"
        )
        if os.path.exists(prev_results_path):
            try:
                with open(prev_results_path, 'r') as f:
                    prev_results = json.load(f)
            except Exception as e:
                print(f"Failed to load previous round results: {e}")

    # Plot based on experiment type
    if server.experiment_type == "n_cmapss":
        # RUL prediction metrics comparison
        metrics = ['rmse', 'mae', 'r_squared', 'within_10_cycles', 'within_20_cycles']
        titles = ['RMSE (lower is better)', 'MAE (lower is better)',
                 'R² (higher is better)', 'Within ±10 cycles %', 'Within ±20 cycles %']

        plt.figure(figsize=(15, 12))

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(3, 2, i+1)

            # Extract metric values for each track
            track_names = list(track_results.keys())
            metric_values = [track_results[track]['rmse'] if metric == 'rmse' else track_results[track][metric]
                             for track in track_names]

            # Create bar chart
            bars = plt.bar(track_names, metric_values)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

            plt.title(title)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        plt.suptitle(f'Track Performance Comparison - Round {round_num}', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'track_comparison_rul_round_{round_num}.png'),
                    bbox_inches='tight')
        plt.close()

    elif server.experiment_type == "mnist":
        # Classification metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        # If disagreements have expired and we have previous results,
        # Create a special plot showing the transition
        if is_global_only and prev_results and len(prev_results) > 1:
            plt.figure(figsize=(12, 10))

            # Get last round's tracks and current global track
            tracks_to_compare = list(prev_results.keys())

            # Ensure global is first if it exists
            if "global" in tracks_to_compare:
                tracks_to_compare.remove("global")
                tracks_to_compare = ["global"] + tracks_to_compare

            # Create a plot comparing previous round tracks with current global
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                plt.subplot(2, 2, i+1)

                # Extract values for comparison
                previous_values = [prev_results[track][metric] for track in tracks_to_compare]
                current_value = track_results["global"][metric]

                # Create a new list with previous tracks and current global
                all_tracks = tracks_to_compare + ["global (current)"]
                all_values = previous_values + [current_value]

                # Create bar chart
                bars = plt.bar(all_tracks, all_values)

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate(f'{height:.4f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

                plt.title(title)
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1.1)  # Scale for classification metrics
                plt.grid(axis='y')
                plt.tight_layout()

            plt.suptitle(f'Track Performance Comparison - Round {round_num} (Disagreements Expired)', y=1.02, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'track_comparison_mnist_round_{round_num}.png'),
                       bbox_inches='tight')
            plt.close()

        # Standard comparison plot
        else:
            plt.figure(figsize=(12, 10))

            for i, (metric, title) in enumerate(zip(metrics, titles)):
                plt.subplot(2, 2, i+1)

                # Extract metric values for each track
                track_names = list(track_results.keys())
                metric_values = [track_results[track][metric] for track in track_names]

                # Create bar chart
                bars = plt.bar(track_names, metric_values)

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate(f'{height:.4f}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom')

                plt.title(title)
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1.1)  # Scale for classification metrics
                plt.grid(axis='y')
                plt.tight_layout()

            subtitle = "Disagreements Expired - All Clients on Global Track" if is_global_only else ""
            plt.suptitle(f'Track Performance Comparison - Round {round_num}\n{subtitle}', y=1.02, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'track_comparison_mnist_round_{round_num}.png'),
                       bbox_inches='tight')
            plt.close()

def plot_track_progress(server, round_num):
    """Plot track performance across rounds.

    Args:
        server: FederatedServer instance
        round_num: Current round number
    """
    # Check if we have track results
    if "track_results" not in server.training_history or not server.training_history["track_results"]:
        print("No track results found in training history - cannot create progress plots")
        return

    # Get track results from all rounds
    track_history = server.training_history["track_results"]

    # Only proceed if we have at least 2 rounds of track data
    if len(track_history) < 2:
        print(f"Only {len(track_history)} rounds of track data found - need at least 2 to create progress plots")
        return

    print(f"Creating track progress plots with data from {len(track_history)} rounds")

    # Plot directory
    plots_dir = os.path.join(server.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get all rounds with track data
    rounds = sorted([int(r) for r in track_history.keys()])

    # Find all track names across all rounds
    all_tracks = set()
    for r in rounds:
        if str(r) in track_history:
            all_tracks.update(track_history[str(r)].keys())

    all_tracks = sorted(list(all_tracks))

    # Check if we have any track data for current round
    # If not and we're past round 2, check if disagreements have expired
    structure = server._get_structure_config()
    current_tracks_dir = os.path.join(
        server.results_dir,
        structure["round_template"].format(round=round_num),
        "tracks"
    )

    disagreements_expired = False
    if round_num > 2 and not os.path.exists(current_tracks_dir):
        # The disagreements likely expired, we should continue using global track
        disagreements_expired = True
        print(f"No tracks directory for round {round_num} - disagreements likely expired")

    # Define metrics based on experiment type
    if server.experiment_type == "mnist":
        metrics = [
            {"name": "accuracy", "title": "Accuracy"},
            {"name": "precision", "title": "Precision"},
            {"name": "recall", "title": "Recall"},
            {"name": "f1", "title": "F1 Score"}
        ]
    elif server.experiment_type == "n_cmapss":
        metrics = [
            {"name": "rmse", "title": "RMSE"},
            {"name": "mae", "title": "MAE"},
            {"name": "r_squared", "title": "R²"},
            {"name": "within_10_cycles", "title": "Within ±10 cycles %"},
            {"name": "within_20_cycles", "title": "Within ±20 cycles %"}
        ]
    else:
        print(f"Unknown experiment type: {server.experiment_type}")
        return

    # Create a figure for each metric
    for metric_info in metrics:
        plt.figure(figsize=(12, 6))

        metric = metric_info["name"]
        title = metric_info["title"]

        # For each track, plot its metric over time
        for track in all_tracks:
            track_values = []
            valid_rounds = []

            # Collect metric values across rounds for this track
            for r in rounds:
                r_str = str(r)
                if r_str in track_history and track in track_history[r_str]:
                    try:
                        # Some tracks might be missing in certain rounds
                        track_values.append(track_history[r_str][track][metric])
                        valid_rounds.append(r)
                    except KeyError:
                        continue

            # Only plot if we have data
            if valid_rounds and track_values:
                # If disagreements expired, extend the last value to the current round
                # But only for tracks that were active in the last round with tracks
                if disagreements_expired and valid_rounds and valid_rounds[-1] < round_num and track == 'global':
                    # Add a point for the current round with the same value as the last round
                    valid_rounds.append(round_num)
                    track_values.append(track_values[-1])

                plt.plot(valid_rounds, track_values, marker='o', label=track)

        # Add global model metric if available
        if server.experiment_type == "mnist" and metric == "accuracy" and len(server.training_history.get("global_test_accuracy", [])) > 0:
            # Only add Global Model line if 'global' track doesn't already exist in all_tracks
            if 'global' not in all_tracks:
                # Filter only rounds that match track rounds
                global_values = []
                for i, r in enumerate(server.training_history["rounds"]):
                    if r in rounds:
                        global_values.append(server.training_history["global_test_accuracy"][i])

                if global_values:
                    plt.plot(rounds[:len(global_values)], global_values, marker='s', linestyle='--',
                             color='black', linewidth=2, label='Global Model')

        elif server.experiment_type == "n_cmapss" and metric == "rmse" and len(server.training_history.get("global_test_loss", [])) > 0:
            # Only add Global Model line if 'global' track doesn't already exist in all_tracks
            if 'global' not in all_tracks:
                # Filter only rounds that match track rounds
                global_values = []
                for i, r in enumerate(server.training_history["rounds"]):
                    if r in rounds:
                        global_values.append(server.training_history["global_test_loss"][i])

                if global_values:
                    plt.plot(rounds[:len(global_values)], global_values, marker='s', linestyle='--',
                             color='black', linewidth=2, label='Global Model')

        plt.title(f'Track {title} Over Rounds')
        plt.xlabel('Round')
        plt.ylabel(title)
        plt.grid(True)
        plt.legend(loc='best')

        # Save figure
        plt.savefig(os.path.join(plots_dir, f'track_progress_{metric}_round_{round_num}.png'),
                   bbox_inches='tight')
        plt.close()

    # Create a comprehensive multi-metric plot for comparison
    plt.figure(figsize=(15, 10))

    # Different subplot layout based on number of metrics
    rows = 2 if len(metrics) <= 4 else 3
    cols = 2 if len(metrics) <= 4 else (3 if len(metrics) <= 9 else 4)

    for i, metric_info in enumerate(metrics[:rows*cols]):  # Limit to fit subplot grid
        plt.subplot(rows, cols, i+1)

        metric = metric_info["name"]
        title = metric_info["title"]

        # For each track, plot its metric over time
        for track in all_tracks:
            track_values = []
            valid_rounds = []

            # Collect metric values across rounds for this track
            for r in rounds:
                r_str = str(r)
                if r_str in track_history and track in track_history[r_str]:
                    try:
                        track_values.append(track_history[r_str][track][metric])
                        valid_rounds.append(r)
                    except KeyError:
                        continue

            # Only plot if we have data
            if valid_rounds and track_values:
                # If disagreements expired, extend the last value to the current round
                # But only for tracks that were active in the last round with tracks
                if disagreements_expired and valid_rounds and valid_rounds[-1] < round_num and track == 'global':
                    # Add a point for the current round with the same value as the last round
                    valid_rounds.append(round_num)
                    track_values.append(track_values[-1])

                plt.plot(valid_rounds, track_values, marker='o', label=track)

        plt.title(title)
        plt.xlabel('Round')
        plt.ylabel(title)
        plt.grid(True)

        # Only add legend to the first subplot to save space
        if i == 0:
            plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'track_metrics_comparison_round_{round_num}.png'),
               bbox_inches='tight')
    plt.close()

    print(f"Saved track progress plots for round {round_num}")
