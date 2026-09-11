# Dataset Splitting & Baseline

## What this does

Splits the Rice dataset into train, validation, and test sets, then creates a majority-class baseline. Includes a checklist for catching common data leakage issues.

## Problems/challenge this solves

* Creates reproducible train/validation/test splits.
* Establishes a baseline before training a real model.
* Provides a quick checklist for common sources of data leakage.

## Key decisions

* Used an 80/10/10 split with a fixed random seed.
* Calculated the baseline using **only the training set**.
* Kept leakage checks separate in `leakage.md` for reuse.

## How to run

```bash
pip install pandas
python splitter.py
python baseline.py
```
