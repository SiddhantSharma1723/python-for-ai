# Student Performance ML Pipeline

## What this does

Builds a Logistic Regression pipeline to predict whether a student passes using numerical and categorical data, with automated preprocessing and feature engineering.

## Dataset

A small student performance dataset with **15 rows** and five columns: `Age`, `City`, `Study_Hours`, `Internet_Access`, and `Passed`. It also contains missing values to test the preprocessing pipeline.

## Problems/challenge this solves

* Handles missing numerical and categorical values.
* Encodes categorical features and scales numerical features.
* Creates a study-hours-per-age feature.
* Keeps preprocessing and modeling together in one pipeline.
* Saves and reloads the trained pipeline with Joblib.

## Key decisions

* Used `ColumnTransformer` for separate numerical and categorical preprocessing.
* Used `Pipeline` to keep feature engineering, preprocessing, and modeling together.
* Used `handle_unknown="ignore"` for unseen categorical values.
* Verified that the saved pipeline produced the same predictions after reloading.

## How to run

```bash
pip install pandas scikit-learn joblib
python your_script_name.py
```

## Results

**Dataset split:** 15 total rows → 12 train / 1 validation / 2 test

| Metric    | Validation |   Test |
| --------- | ---------: | -----: |
| Accuracy  |       100% |    50% |
| Precision |       100% |    50% |
| Recall    |       100% |   100% |
| F1        |       100% | 66.67% |

The validation and test sets are extremely small, so these metrics have high variance. A single prediction changes the score substantially, meaning they should **not be treated as reliable evidence of generalization performance**.
