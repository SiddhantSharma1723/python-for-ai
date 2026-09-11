# House Price Linear Regression

## What this does

Trains a Linear Regression model to predict house prices and compares it against a baseline using MAE and RMSE.

## Problems/challenge this solves

* Predicts house prices from multiple features.
* Checks whether the model actually improves on the baseline.
* Compares training and validation errors to spot potential overfitting.
* Uses residuals to inspect prediction errors.

## Key decisions

* Used the existing train/validation/test split and baseline.
* Evaluated the model with both MAE and RMSE.
* Used the validation set for model evaluation rather than the test set.
* Compared training and validation errors to measure the generalization gap.

## How to run

```bash
pip install pandas matplotlib scikit-learn
python your_script_name.py
```
