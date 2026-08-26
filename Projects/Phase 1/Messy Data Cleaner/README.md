# Student Records Data Cleaning

## What this does

Cleans and validates messy student records using Pandas, including data type conversion, missing-value handling, standardization, date parsing, and duplicate removal.

## Problems/challenge this solves

* Handles invalid and missing `age` and `score` values.
* Standardizes inconsistent gender and city values.
* Converts mixed-format dates into proper datetime values.
* Removes duplicate records.
* Validates the cleaned dataset with automated checks.

## Key decisions

* Used `errors="coerce"` to safely handle invalid numeric/date values.
* Mapped inconsistent categorical values to standardized labels.
* Added schema assertions to verify data quality after cleaning.

## How to run

```bash
pip install pandas
python your_script_name.py
```
