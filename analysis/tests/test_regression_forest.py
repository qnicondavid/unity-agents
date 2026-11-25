import random
import pytest
from analysis.forest.regression import (
    variance,
    best_split_point_fast,
    build_tree,
    build_forest,
    predict_forest,
)


def test_variance_calculation():
    targets = [1.0, 2.0, 3.0]
    v = variance(targets)
    assert v > 0
    assert v < 1.0


def test_variance_identical_values():
    targets = [5.0, 5.0, 5.0]
    assert variance(targets) == 0.0


def test_split_point_selection():
    values = [1.0, 2.0, 10.0, 11.0]
    targets = [1.0, 1.0, 10.0, 10.0]
    split, gain = best_split_point_fast(values, targets, min_leaf=1)
    
    assert split is not None
    assert gain > 0
    assert 2.0 <= split <= 10.0


def test_tree_building():
    rows = [[1.0], [2.0], [10.0], [11.0]]
    targets = [1.0, 1.0, 10.0, 10.0]
    
    tree = build_tree(rows, targets, [0], m=1, rng=random.Random(1),
                     max_depth=5, min_leaf=1)
    
    assert isinstance(tree, dict)
    assert "attribute" in tree or "label" in tree


def test_tree_max_depth():
    rows = [[float(i)] for i in range(10)]
    targets = [float(i) for i in range(10)]
    
    tree = build_tree(rows, targets, [0], m=1, rng=random.Random(1),
                     max_depth=1, min_leaf=1)
    
    def get_depth(node, depth=0):
        if "label" in node:
            return depth
        return max(get_depth(node["below"], depth + 1),
                  get_depth(node["above"], depth + 1))
    
    assert get_depth(tree) <= 2


def test_forest_creation():
    rows = [[float(i)] for i in range(20)]
    targets = [float(i) * 2.0 for i in range(20)]
    
    forest = build_forest(rows, targets, n_trees=5, m=1, max_depth=5,
                         min_leaf=1, sampling="split", seed=1)
    
    assert len(forest) == 5
    assert all(isinstance(tree, dict) for tree in forest)


def test_forest_prediction():
    rows = [[float(i)] for i in range(20)]
    targets = [float(i) * 2.0 for i in range(20)]
    
    forest = build_forest(rows, targets, n_trees=10, m=1, max_depth=5,
                         min_leaf=1, sampling="split", seed=1)
    
    pred = predict_forest(forest, [5.0])
    assert isinstance(pred, (int, float))
    assert 5.0 < pred < 15.0


def test_forest_reproduces_with_seed():
    rows = [[float(i), float(i*0.5)] for i in range(20)]
    targets = [float(i) for i in range(20)]
    
    f1 = build_forest(rows, targets, n_trees=5, m=2, max_depth=5,
                     min_leaf=1, sampling="split", seed=1)
    f2 = build_forest(rows, targets, n_trees=5, m=2, max_depth=5,
                     min_leaf=1, sampling="split", seed=1)
    
    assert len(f1) == len(f2)


def test_forest_monotonic_on_linear_data():
    rows = [[float(i)] for i in range(20)]
    targets = [float(i) * 2.5 + 1.0 for i in range(20)]
    
    forest = build_forest(rows, targets, n_trees=15, m=1, max_depth=10,
                         min_leaf=1, sampling="split", seed=1)
    
    pred_low = predict_forest(forest, [2.0])
    pred_mid = predict_forest(forest, [10.0])
    pred_high = predict_forest(forest, [18.0])
    
    assert pred_low < pred_mid < pred_high


def test_forest_handles_multivariate():
    rows = [[float(i), float(i*0.5)] for i in range(20)]
    targets = [float(i) * 1.5 for i in range(20)]
    
    forest = build_forest(rows, targets, n_trees=10, m=2, max_depth=5,
                         min_leaf=1, sampling="split", seed=1)
    
    pred1 = predict_forest(forest, [5.0, 2.5])
    pred2 = predict_forest(forest, [15.0, 7.5])
    
    assert isinstance(pred1, (int, float))
    assert isinstance(pred2, (int, float))
    assert pred1 < pred2
