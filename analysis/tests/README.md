# Tests

Tests to verify the regression forest and MLP models can learn to predict reward thresholds from training steps.

## Running

From the repository root, install pytest in the same Python environment as PyTorch:

```bash
python -m pip install pytest
```

```bash
python -m pytest analysis/tests/ -v
python -m pytest analysis/tests/test_regression_forest.py -v
python -m pytest analysis/tests/test_mlp.py -v
```

## What's Tested

1. Regression Forest - Variance calculation, split selection, tree/forest construction, predictions, determinism with seeds, and multivariate data.

2. MLP - Data scaling, network architecture, loss computation, training, predictions, learning on linear data, and validation.

