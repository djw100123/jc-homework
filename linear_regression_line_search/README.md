# Jiangcai

This repository contains a NumPy implementation of linear regression for the UCI Wine Quality dataset.

## Files

- `linear_regression_line_search.py`: linear regression with backtracking line search
- `linear_regression_gd_uci.py`: standard gradient descent version
- `winequality-red.csv`: dataset used by the scripts

## Requirements

- Python 3.10+
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python linear_regression_line_search.py
```

or

```bash
python linear_regression_gd_uci.py
```

## Notes

- The scripts will try to download `winequality-red.csv` automatically if it is missing.
- The dataset is stored in the repository root so the scripts can run without path changes.
