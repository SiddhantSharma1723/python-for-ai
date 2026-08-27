# Pokemon Stats Data Cleaning

## What this does

Cleans and analyzes messy Pokemon stats data using Pandas and Matplotlib, including 
numeric type conversion, missing-value handling, categorical standardization, 
outlier detection, and exploratory visualization.

## Problems/challenges this solves

* Fixes HP values stored as inconsistent strings (some numbers, some with 
  trailing whitespace) by coercing to numeric.😉
* Standardizes three different "missing" representations in Type2 
  (blank, NaN, and the literal string "None") into a single "Unknown" label — 
  since a missing Type2 usually just means the Pokemon has only one type, 
  not corrupted data.
* Normalizes inconsistent casing in Type1 (e.g. "FIRE" vs "Fire").
* Removes duplicate rows.
* Flags and removes outliers in Attack using the IQR method.

## Key decisions

* Used `errors="coerce"` on HP to safely convert bad values to NaN instead 
  of crashing.
* Treated missing Type2 as "Unknown" rather than dropping those rows — 
  a missing second type is valid data (single-type Pokemon), not an error.
* Used the IQR method (1.5× IQR beyond Q1/Q3) to detect Attack outliers, 
  rather than an arbitrary cutoff.

## Visualizations

* `attack_distribution.png` — histogram of Attack values
* `defense_by_type1.png` — box plot of Defense grouped by Type1
* `defense_by_legendary.png` — box plot of Defense grouped by Legendary status
* `attack_vs_defense.png` — scatter matrix of Attack vs Defense
* `speed_vs_spatk.png` — scatter matrix of Speed vs Sp_Atk

## Findings

The raw data mixed several inconsistent representations of missing/invalid 
values in the same file (blank, NaN, string "None", and malformed numeric 
strings) — a realistic example of what real-world messy data looks like 
before cleaning.

## How to run

\`\`\`bash
pip install pandas matplotlib
python clean_pokemon.py
\`\`\`

Outputs `pokemon_stats_clean.csv` and 5 PNG plots in the same folder.