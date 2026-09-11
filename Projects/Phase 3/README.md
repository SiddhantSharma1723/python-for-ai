# Linear & Logistic Regression From Scratch

## What this does

Implements linear regression and logistic regression from scratch using only 
NumPy — no scikit-learn — including manual gradient derivation and gradient 
descent optimization.

## Problems/challenges this solves

* Fits a line (`y = wx + b`) to noisy synthetic data via gradient descent.
* Classifies binary labels using logistic regression with a sigmoid 
  activation and cross-entropy loss.
* Demonstrates how learning rate and epoch count affect convergence 
  (including a first run that diverged at lr=0.1, fixed by lowering lr).

## Key decisions

* Derived the loss gradients (∂Loss/∂w, ∂Loss/∂b) by hand before coding them, 
  for both MSE (linear) and binary cross-entropy (logistic).
* Chose a small learning rate (0.001–0.01) after an initial run at 0.1 
  diverged — larger `x` values amplify the gradient for `w`, requiring a 
  smaller step size to stay stable.
* Increased epoch count when `b` lagged behind `w` in convergence — since 
  `w`'s gradient scales with `x` and `b`'s doesn't, they converge at 
  different speeds.

## Results

* Linear regression: recovered w ≈ 5.07, b ≈ 4.79 (true values: w=5, b=5)
* Logistic regression: recovered w ≈ 0.30, b ≈ -0.80, correctly separating 
  classes around the true decision boundary at x=5

## How to run

\`\`\`bash
pip install numpy matplotlib
python regression_from_scratch.py
\`\`\`

Outputs loss curves and fitted-line/decision-boundary plots.