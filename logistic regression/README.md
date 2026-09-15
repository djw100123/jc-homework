# Regularized Logistic Regression with Gradient Descent

This project implements binary logistic regression from scratch with NumPy.
It uses the UCI Bank Marketing dataset and includes L2 regularization,
gradient descent, one-hot encoding, and classification metrics.

## Project Contents

```text
regularized_logistic_regression_gd.py
    Main program. Downloads the dataset when necessary, preprocesses the data,
    trains the model, and prints evaluation metrics.

作业2_正则化逻辑回归梯度下降思路.txt
    Explanation of the assignment, model, gradient descent, regularization,
    and generalization.

regularized_logistic_regression_gd代码解读.txt
    Function-by-function explanation of the Python program.

补充说明_one-hot编码.txt
    Explanation of one-hot encoding.

补充说明_分层划分与多次随机实验.txt
    Explanation of stratified splitting and repeated experiments.

requirements.txt
    Python dependency list.
```

## Requirements

- Python 3.10 or newer
- NumPy
- Internet access on the first run if `bank-full.csv` is not already present

Install the dependency:

```bash
python -m pip install -r requirements.txt
```

## Run

Run the program from this folder:

```bash
python regularized_logistic_regression_gd.py
```

On the first run, the program downloads the UCI Bank Marketing archive,
extracts `bank-full.csv`, preprocesses the data, and trains the model.
Later runs reuse the local dataset.

## Implementation

The program performs the following steps:

1. Loads the UCI Bank Marketing dataset.
2. Converts the target `y` from `yes`/`no` to `1`/`0`.
3. Uses a stratified train/test split.
4. Standardizes numeric features using training-set statistics.
5. Converts categorical features with one-hot encoding.
6. Adds a bias column.
7. Trains logistic regression with batch gradient descent.
8. Adds L2 regularization to non-bias weights.
9. Reports accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

The model is implemented directly with NumPy. It does not use
`sklearn.linear_model.LogisticRegression` for training.

## Dataset

UCI Bank Marketing:

https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset is used for educational purposes. The target variable indicates
whether a client subscribed to a term deposit.

## Example Output

```text
Dataset: UCI Bank Marketing
Training samples: 36170
Testing samples:  9041
Encoded features: 583
Final training loss: 0.269403

Testing metrics
  Accuracy:  0.8955
  Precision: 0.6505
  Recall:    0.2289
  F1-score:  0.3387
  ROC-AUC:   0.8973
```

The exact result can vary if the implementation parameters or dataset split
are changed.

## Files Not Required in GitHub

The following local files should not be uploaded because the program can
download the dataset automatically:

```text
bank-full.csv
bank-full整理版.csv
bank+marketing.zip
```

Python cache folders such as `__pycache__` should also be excluded.
