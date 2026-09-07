"""
Linear regression with NumPy and line search gradient descent.

Dataset: UCI Wine Quality, red wine subset
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

This version uses backtracking line search to choose the learning rate
automatically during each gradient descent iteration.
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
DATA_FILE = Path("winequality-red.csv")
TARGET_COLUMN = "quality"


def download_dataset(data_file):
    if data_file.exists():
        return

    print(f"Downloading dataset to {data_file} ...")
    try:
        urlretrieve(DATA_URL, data_file)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Download it manually from:\n"
            f"{DATA_URL}\n"
            f"and save it as {data_file.resolve()}"
        ) from exc


def load_wine_quality_data(data_file):
    download_dataset(data_file)

    with data_file.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()

    delimiter = "," if "," in first_line and ";" not in first_line else ";"
    header = first_line.replace('"', "").split(delimiter)

    data = np.genfromtxt(
        data_file,
        delimiter=delimiter,
        skip_header=1,
        dtype=float,
    )

    target_index = header.index(TARGET_COLUMN)
    x = np.delete(data, target_index, axis=1)
    y = data[:, target_index]
    feature_names = [name for i, name in enumerate(header) if i != target_index]
    return x, y, feature_names


def train_test_split(x, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    test_size = int(len(x) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def add_bias_column(x):
    return np.c_[np.ones(x.shape[0]), x]


def mse(y_true, y_pred):
    errors = y_pred - y_true
    return np.mean(errors ** 2)


def loss_value(x, y, weights):
    predictions = x @ weights
    return mse(y, predictions)


def loss_gradient(x, y, weights):
    n_samples = x.shape[0]
    predictions = x @ weights
    errors = predictions - y
    return (2 / n_samples) * (x.T @ errors)


def backtracking_line_search(
    x,
    y,
    weights,
    gradient,
    direction,
    initial_step=1.0,
    shrink=0.5,
    c=1e-4,
    max_attempts=50,
):
    current_loss = loss_value(x, y, weights)
    step_size = initial_step
    slope = gradient @ direction

    for _ in range(max_attempts):
        new_weights = weights + step_size * direction
        new_loss = loss_value(x, y, new_weights)

        if new_loss <= current_loss + c * step_size * slope:
            return step_size

        step_size *= shrink

    return step_size


def gradient_descent_with_line_search(x, y, epochs=5000, tolerance=1e-8):
    weights = np.random.standard_normal(x.shape[1])
    losses = []
    step_sizes = []

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
            initial_step=1.0,
            shrink=0.5,
            c=1e-4,
        )

        weights = weights + step_size * direction
        current_loss = loss_value(x, y, weights)
        step_sizes.append(step_size)

        if epoch % 500 == 0 or epoch == epochs - 1:
            losses.append((epoch, current_loss, step_size))

    return weights, losses, step_sizes


def evaluate_model(x_train, y_train, x_test, y_test, weights):
    train_predictions = x_train @ weights
    test_predictions = x_test @ weights

    return {
        "train_mse": mse(y_train, train_predictions),
        "test_mse": mse(y_test, test_predictions),
        "sample_predictions": test_predictions[:10],
        "sample_actual": y_test[:10],
    }


def main():
    x, y, feature_names = load_wine_quality_data(DATA_FILE)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_train, x_test = standardize(x_train, x_test)
    x_train = add_bias_column(x_train)
    x_test = add_bias_column(x_test)

    weights, losses, step_sizes = gradient_descent_with_line_search(
        x_train,
        y_train,
        epochs=5000,
    )
    results = evaluate_model(x_train, y_train, x_test, y_test, weights)

    print("Dataset: UCI Wine Quality - red wine")
    print(f"Samples: {len(x)}")
    print(f"Features: {', '.join(feature_names)}")
    print()

    print("Gradient descent with line search progress:")
    for epoch, loss, step_size in losses:
        print(
            f"  Epoch {epoch:4d}: "
            f"train MSE = {loss:.4f}, step size = {step_size:.6f}"
        )
    print()

    print("Final evaluation:")
    print(f"  Train MSE: {results['train_mse']:.4f}")
    print(f"  Test MSE:  {results['test_mse']:.4f}")
    print(f"  Average step size: {np.mean(step_sizes):.6f}")
    print()

    print("First 10 test predictions:")
    for predicted, actual in zip(results["sample_predictions"], results["sample_actual"]):
        print(f"  predicted = {predicted:.2f}, actual = {actual:.0f}")

    print()
    print("Learned parameters:")
    print(f"  bias: {weights[0]:.4f}")
    for name, weight in zip(feature_names, weights[1:]):
        print(f"  {name}: {weight:.4f}")


if __name__ == "__main__":
    main()
