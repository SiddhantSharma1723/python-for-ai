# JEE Pass Prediction

## What this does

Builds a Logistic Regression classifier to predict whether a student will pass based on demographic, study, and internet-access features. The project uses feature engineering, preprocessing pipelines, probability threshold analysis, validation error analysis, and model persistence.

## Dataset

The dataset contains 2,016 student records with a balanced target: 1,008 `Passed = 0` and 1,008 `Passed = 1`.

Input features include `Age`, `City`, `Study_Hours`, and `Internet_Access`. A `Study_hours_per_age` feature is created during the pipeline.

## Problems/challenge this solves

* Handles missing numerical and categorical values within the preprocessing pipeline.
* Converts categorical features into a model-ready representation using one-hot encoding.
* Adds a derived feature without modifying the original dataset.
* Evaluates how changing the classification threshold affects precision, recall, and F1.
* Analyzes validation errors instead of relying only on aggregate metrics.
* Saves and reloads the complete trained pipeline for reuse.

## Key decisions

* Used a single `Pipeline` with `ColumnTransformer` so feature engineering and preprocessing are applied consistently during training and prediction.
* Used mean imputation and standardization for numerical features.
* Used most-frequent imputation and one-hot encoding for categorical features, with unknown categories handled safely.
* Used Logistic Regression as the classifier.
* Evaluated thresholds from `0.50` to `0.95` on the validation set and selected `0.50` as the final threshold.
* Kept the test set separate for the final evaluation.
* Saved the complete pipeline with `joblib` and verified that the reloaded pipeline produced the same validation predictions.

## How to run

```bash
jee_pass_prediction.py
```

## Results

Validation:

* Accuracy: **99.60%**
* Precision: **99.37%**
* Recall: **100.00%**
* F1: **99.68%**

Test:

* Accuracy: **100.00%**
* Precision: **100.00%**
* Recall: **100.00%**
* F1: **100.00%**

Threshold analysis found the best validation F1 at `0.50` and `0.55`, both with **99.68% F1** and **100% recall**. The selected threshold was `0.50`.

The validation set contained **one false positive** and no false negatives. The incorrect prediction had a model probability of approximately **0.877**, showing that high confidence does not guarantee a correct prediction.
