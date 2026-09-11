# Student Pass Prediction

## What this does

Trains a Logistic Regression model to predict whether a student passes based on study habits and attendance, then evaluates it using classification metrics.

## Problems/challenge this solves

* Predicts student pass/fail outcomes.
* Standardizes numerical features before training.
* Evaluates the model using accuracy, precision, recall, F1, and ROC-AUC.

## Key decisions

* Used `StandardScaler` because Logistic Regression can be sensitive to feature scale.
* Fit the scaler only on training data and applied it to validation data.
* Used predicted probabilities for ROC-AUC evaluation.

## How to run

```bash
pip install pandas scikit-learn
python your_script_name.py
```
