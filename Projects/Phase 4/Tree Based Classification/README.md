# Tree-Based Classification Model Comparison

## What this does

Compares Decision Tree, Random Forest, and Gradient Boosting models for predicting student pass/fail outcomes.

## Problems/challenge this solves

* Tests different tree-based approaches on the same dataset.
* Compares training, validation, and test performance.
* Evaluates models using accuracy, precision, recall, and F1 score.
* Identifies the strongest model among the approaches tested.

## Key decisions

* Used the same features and train/validation/test split for each model.
* Limited the Decision Tree depth to reduce overfitting.
* Used 200 trees and `min_samples_leaf=10` for the Random Forest.
* Compared Gradient Boosting against the other tree-based models before selecting a model.

## How to run

```bash
pip install pandas scikit-learn
python your_script_name.py
```
