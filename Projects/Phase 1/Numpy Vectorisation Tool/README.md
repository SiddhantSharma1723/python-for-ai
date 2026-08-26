# NumPy Vectorization vs Python Loops

## What this does

Compares Python loops with NumPy vectorized operations for statistical analysis on 1 million randomly generated scores.

## Problems/challenge this solves

* Calculates mean, standard deviation, and Z-scores using two approaches.
* Verifies that both implementations produce equivalent results.
* Benchmarks their execution time.
* Demonstrates the performance advantage of NumPy vectorization.

## Key decisions

* Used `np.isclose()` and `np.allclose()` to validate numerical results.
* Used NumPy vectorization for array-wide calculations.
* Used `np.where()` to identify scores with Z-scores above 2.

## How to run

```bash
pip install numpy
python your_script_name.py
```
