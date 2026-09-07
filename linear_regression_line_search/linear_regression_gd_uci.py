"""
Linear regression with NumPy on a UCI dataset.

Dataset: UCI Wine Quality, red wine subset
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

The script downloads the CSV if it is not already present, trains a linear
regression model with gradient descent, and reports MSE on train/test data.
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


def gradient_descent(x, y, learning_rate=0.3, epochs=5000):
    weights = np.random.standard_normal(x.shape[1])
    losses = []
    n_samples = x.shape[0]

    for epoch in range(epochs):
        predictions = x @ weights
        errors = predictions - y
        gradient = (2 / n_samples) * (x.T @ errors)
        #它是在一次性算出：每个参数的梯度，每个参数该怎么改
        weights -= learning_rate * gradient
        #errors：现在预测错了多少
        #x.T @ errors：每个特征到底“带来了多少错误”
        #gradient：每个参数应该怎么改

        if epoch % 500 == 0 or epoch == epochs - 1:
            losses.append((epoch, mse(y, predictions)))

    return weights, losses


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


    weights, losses = gradient_descent(
        x_train,
        y_train,
        learning_rate=0.3,
        epochs=5000,
    )
    results = evaluate_model(x_train, y_train, x_test, y_test, weights)

    print("Dataset: UCI Wine Quality - red wine")
    print(f"Samples: {len(x)}")
    print(f"Features: {', '.join(feature_names)}")
    print()

    print("Gradient descent progress:")
    for epoch, loss in losses:
        print(f"  Epoch {epoch:4d}: train MSE = {loss:.4f}")
    print()

    print("Final evaluation:")
    print(f"  Train MSE: {results['train_mse']:.4f}")
    print(f"  Test MSE:  {results['test_mse']:.4f}")
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
