# Gradient Boosting Calibration

## What this does

Trains a Gradient Boosting classifier with cross-validated hyperparameter tuning, then calibrates its predicted probabilities using sigmoid calibration. The project evaluates both classification performance and probability quality using calibration curves and Brier scores.

## Dataset

The dataset contains 2,000 student records with four numerical features: `hours_studied`, `attendance_pct`, `prev_score`, and `sleep_hours`. The target is `passed`, with 1,200 passing and 800 failing examples.

The data is split into 1,600 training, 200 validation, and 200 test examples.

## Problems/challenge this solves

* Finds Gradient Boosting hyperparameters using stratified 5-fold cross-validation.
* Evaluates whether the model's predicted probabilities are well calibrated.
* Compares original and sigmoid-calibrated probabilities using calibration curves and Brier scores.
* Keeps the validation and test sets separate from model selection and calibration training.
* Evaluates the final calibrated model on a held-out test set.

## Key decisions

* Used `StratifiedKFold` with 5 folds to preserve class proportions during hyperparameter search.
* Used `GridSearchCV` with accuracy to select the Gradient Boosting configuration.
* Calibrated the selected model with sigmoid calibration using 5-fold cross-validation on the training data.
* Used the validation set to compare probability calibration before evaluating the final model on the test set.
* Used Brier score alongside the calibration curve because classification accuracy alone does not measure the quality of predicted probabilities.

## How to run

```bash
python gradient_boosting_calibration.py
```

## Results

Best Gradient Boosting parameters:

* `learning_rate`: **0.1**
* `max_depth`: **2**
* `n_estimators`: **100**

5-fold cross-validation:

* Mean accuracy: **69.94%**
* Standard deviation: **0.75%**

Validation Brier score:

* Original: **0.1956**
* Sigmoid calibrated: **0.1915**

Test performance of the calibrated model:

* Accuracy: **67.00%**
* Precision: **69.17%**
* Recall: **78.63%**
* F1: **73.60%**
* Brier score: **0.2088**

Sigmoid calibration reduced the validation Brier score, indicating improved probability calibration on the validation set. The final test accuracy was 67%, showing that better-calibrated probabilities did not necessarily result in better classification accuracy.
