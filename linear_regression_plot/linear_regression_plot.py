"""
Plot the iterative refinement process of linear regression gradient descent.

This script reuses the data-processing and line-search functions from
linear_regression_line_search.py, but records the loss at every epoch and
plots the recorded training history.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from linear_regression_line_search import (
    add_bias_column,
    backtracking_line_search,
    load_wine_quality_data,
    loss_gradient,
    loss_value,
    standardize,
    train_test_split,
)


DATA_FILE = Path(__file__).with_name("winequality-red.csv")
PLOT_FILE = Path(__file__).with_name("gradient_descent_loss.png")


def gradient_descent_with_history(x, y, epochs=5000, tolerance=1e-8, seed=42):
    """Run gradient descent and save loss and step size from every epoch."""
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal(x.shape[1])
    history = {
        "epochs": [],
        "losses": [],
        "step_sizes": [],
    }

    for epoch in range(epochs):
        gradient = loss_gradient(x, y, weights)
        direction = -gradient

        if np.linalg.norm(gradient) < tolerance:
            break

        step_size = backtracking_line_search(
            x,
            y,
            weights,
            gradient,
            direction,
        )
        weights = weights + step_size * direction

        history["epochs"].append(epoch)
        history["losses"].append(loss_value(x, y, weights))
        history["step_sizes"].append(step_size)

    return weights, history


def plot_training_history(history, output_file):
    """Plot training loss and the line-search step size, then save the figure."""
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(history["epochs"], history["losses"], color="tab:blue")
    axes[0].set_title("Gradient Descent Iterative Refinement")
    axes[0].set_ylabel("Training MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epochs"], history["step_sizes"], color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Step size")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file.resolve()}")
    plt.show()
    plt.close(figure)


def main():
    x, y, _ = load_wine_quality_data(DATA_FILE)
    x_train, _, y_train, _ = train_test_split(x, y)
    x_train, _ = standardize(x_train, x_train)
    x_train = add_bias_column(x_train)

    weights, history = gradient_descent_with_history(x_train, y_train)
    plot_training_history(history, PLOT_FILE)

    print(f"Epochs recorded: {len(history['epochs'])}")
    print(f"Initial training MSE: {history['losses'][0]:.4f}")
    print(f"Final training MSE: {history['losses'][-1]:.4f}")
    print(f"Parameters learned: {len(weights)}")


if __name__ == "__main__":
    main()
