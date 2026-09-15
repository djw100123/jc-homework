"""
Regularized logistic regression trained with gradient descent.

Dataset: UCI Bank Marketing, bank-full.csv
The dataset is downloaded automatically when it is not already present.
"""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import csv
import io

import numpy as np


DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_DIR = Path(__file__).parent
ZIP_FILE = DATA_DIR / "bank+marketing.zip"
DATA_FILE = DATA_DIR / "bank-full.csv"


def download_dataset():
    if DATA_FILE.exists():
        return

    if not ZIP_FILE.exists():
        print("Downloading UCI Bank Marketing dataset...")
        urlretrieve(DATA_URL, ZIP_FILE)

    with ZipFile(ZIP_FILE) as outer_archive:
        inner_zip_name = next(
            name for name in outer_archive.namelist() if name.endswith("bank.zip")
        )
        inner_zip_data = outer_archive.read(inner_zip_name)

    with ZipFile(io.BytesIO(inner_zip_data)) as inner_archive:
        member = next(
            name for name in inner_archive.namelist() if name.endswith("bank-full.csv")
        )
        with inner_archive.open(member) as source, DATA_FILE.open("wb") as target:
            target.write(source.read())


def load_data():
    download_dataset()

    with DATA_FILE.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter=";"))

    target_name = "y"
    feature_names = [name for name in rows[0] if name != target_name]

    #给特征分类
    categorical_names = [
        name for name in feature_names if not rows[0][name].replace(".", "", 1).isdigit()
    ]
    numeric_names = [
        name for name in feature_names if name not in categorical_names
    ]


    numeric_values = np.array(
        [[float(row[name]) for name in numeric_names] for row in rows],
        dtype=float,
    )
    categorical_values = np.array(
        [[row[name] for name in categorical_names] for row in rows],
        dtype=object,
    )

    labels = np.array([1.0 if row[target_name] == "yes" else 0.0 for row in rows])

    return (
        numeric_values,
        categorical_values,
        labels,
        numeric_names,
        categorical_names,
    )

#先按照 y 的类别把样本分组，再在每个类别内部随机打乱，最后从每个类别里按比例抽一部分进测试集
#防止普通随机划分时，某个类别在训练集或测试集里比例偏差太大，从而导致训练结果或测试结果不稳定。
def stratified_split(n_samples, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)

    train_indices = []
    test_indices = []

    #这个循环会分别处理label = 0.0和label = 1.0
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)

        #把当前类别的样本下标随机打乱
        rng.shuffle(indices)

        test_size = max(1, int(len(indices) * test_ratio))

        test_indices.extend(indices[:test_size])
        train_indices.extend(indices[test_size:])

    #把测试集下标整体打乱
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return np.array(train_indices), np.array(test_indices)


def fit_preprocessor(
    numeric_train,
    categorical_train,
    numeric_names,
    categorical_names,
):

    numeric_mean = numeric_train.mean(axis=0)
    numeric_std = numeric_train.std(axis=0)

    #处理特殊情况
    numeric_std[numeric_std == 0] = 1.0

    categories = {
        name: sorted(

            #用来保存每个类别特征有哪些类别
            #为后续的one-hot编码做准备
            {row[index] for row in categorical_train}
        )

        #遍历所有类别特征
        for index, name in enumerate(categorical_names)
    }

    return numeric_mean, numeric_std, categories


def transform_data(
    numeric_values,
    categorical_values,
    numeric_mean,
    numeric_std,
    categories,
    categorical_names,
):
    numeric_scaled = (numeric_values - numeric_mean) / numeric_std
    encoded_columns = []

    for index, name in enumerate(categorical_names):
        category_to_index = {
            value: category_index
            for category_index, value in enumerate(categories[name])
        }

        #创建一个全 0 矩阵，用来保存当前类别特征的 one-hot 编码结果
        encoded = np.zeros((len(categorical_values), len(categories[name])))

        for row_index, value in enumerate(categorical_values[:, index]):
            if value in category_to_index:
                encoded[row_index, category_to_index[value]] = 1.0

        #将所有非数字的编码后特征矩阵拼接起来
        encoded_columns.append(encoded)

    if encoded_columns:
        return np.hstack([numeric_scaled, *encoded_columns])
    return numeric_scaled


def add_bias_column(x):
    return np.c_[np.ones(x.shape[0]), x]



#sigmoid() 函数把任意实数转换成 0 到 1 之间的数
def sigmoid(z):
    probabilities = np.empty_like(z, dtype=float)
    #先处理大于0
    positive = z >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    #取反处理小于0
    exp_z = np.exp(z[~positive])
    probabilities[~positive] = exp_z / (1.0 + exp_z)

    return probabilities


def loss_value(x, y, weights, regularization_strength):
    #这里的特征矩阵x已经去除了偏置
    scores = x @ weights
    data_loss = np.mean(np.logaddexp(0.0, scores) - y * scores)

    #计算 L2 正则化惩罚项
    penalty = (regularization_strength / 2.0) * np.sum(weights[1:] ** 2)

    return data_loss + penalty


def loss_gradient(x, y, weights, regularization_strength):
    probabilities = sigmoid(x @ weights)

    #计算交叉熵损失梯度
    gradient = (x.T @ (probabilities - y)) / len(y)
    gradient[1:] += regularization_strength * weights[1:]
    return gradient


def train(
    x,
    y,
    learning_rate=0.05,
    regularization_strength=0.01,
    epochs=2000,
    tolerance=1e-7,
    seed=42,
):
    rng = np.random.default_rng(seed)
    weights = np.zeros(x.shape[1], dtype=float)
    losses = []

    for epoch in range(epochs):
        gradient = loss_gradient(x, y, weights, regularization_strength)
        weights -= learning_rate * gradient

        current_loss = loss_value(
            x, y, weights, regularization_strength
        )
        if epoch % 100 == 0 or epoch == epochs - 1:
            losses.append((epoch + 1, current_loss))

        if np.linalg.norm(gradient) < tolerance:
            break

    return weights, losses


def confusion_counts(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return int(tp), int(tn), int(fp), int(fn)


def classification_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tp, tn, fp, fn = confusion_counts(y_true.astype(int), predictions)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc_score(y_true, probabilities):
    positive_count = np.sum(y_true == 1)
    negative_count = np.sum(y_true == 0)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(probabilities)
    sorted_labels = y_true[order]
    ranks = np.arange(1, len(y_true) + 1, dtype=float)
    rank_sum = np.sum(ranks[sorted_labels == 1])
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def evaluate(x, y, weights):
    probabilities = sigmoid(x @ weights)
    return classification_metrics(y, probabilities)


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(
        "  Confusion matrix "
        f"(TN, FP, FN, TP): "
        f"({metrics['tn']}, {metrics['fp']}, {metrics['fn']}, {metrics['tp']})"
    )


def main():
    numeric, categorical, y, numeric_names, categorical_names = load_data()
    train_indices, test_indices = stratified_split(len(y), y)

    numeric_train = numeric[train_indices]
    categorical_train = categorical[train_indices]
    numeric_test = numeric[test_indices]
    categorical_test = categorical[test_indices]

    mean, std, categories = fit_preprocessor(
        numeric_train,
        categorical_train,
        numeric_names,
        categorical_names,
    )
    x_train = transform_data(
        numeric_train, categorical_train, mean, std, categories, categorical_names
    )
    x_test = transform_data(
        numeric_test, categorical_test, mean, std, categories, categorical_names
    )
    x_train = add_bias_column(x_train)
    x_test = add_bias_column(x_test)

    weights, losses = train(
        x_train,
        y[train_indices],
        learning_rate=0.05,
        regularization_strength=0.01,
        epochs=2000,
    )

    print("Dataset: UCI Bank Marketing")
    print(f"Training samples: {len(train_indices)}")
    print(f"Testing samples:  {len(test_indices)}")
    print(f"Encoded features: {x_train.shape[1] - 1}")
    print(f"Final training loss: {losses[-1][1]:.6f}")
    print_metrics("Training metrics", evaluate(x_train, y[train_indices], weights))
    print_metrics("Testing metrics", evaluate(x_test, y[test_indices], weights))


if __name__ == "__main__":
    main()
