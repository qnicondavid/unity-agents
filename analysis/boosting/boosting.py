# Gradient boosting for regression with the squared error, as in Algorithm 10.3 of
# Hastie, Tibshirani and Friedman (2009), The Elements of Statistical Learning, 2nd ed.,
# Section 10.10, with the shrinkage of Section 10.12.1 and the subsampling of
# Friedman (2002), "Stochastic gradient boosting".
#
# The trees are the regression trees of our regression forest (analysis/forest/regression.py),
# built on all attributes and without a bootstrap. What changes from the forest:
# - the trees are built one after the other, not independently
# - each tree is fitted to what the trees before it got wrong (the residuals), not to the targets
# - the prediction is the sum of the trees, each scaled down by the learning rate, not their mean
# - the trees are shallow on purpose: a forest averages deep trees to lower their variance,
#   boosting adds small trees to lower the bias step by step

import random

from analysis.forest.forest import predict
from analysis.forest.regression import build_tree


# Squared error loss  L(y, F(x)) = (y - F(x))^2 / 2  [ESL, Table 10.2]
# - its negative gradient with respect to F(x) is y - F(x), the residual
# - so each tree learns the part of the target the model so far has not explained
def residuals(targets, fitted):
    return [target - value for target, value in zip(targets, fitted)]


# Stochastic gradient boosting  [Friedman 2002; ESL, Section 10.12.2]
# - each tree sees a random fraction of the training runs, drawn without replacement
# - with a fraction of 1, every tree sees all runs and nothing is random
def draw_runs(n, fraction, rng):
    if fraction >= 1:
        return list(range(n))
    return sorted(rng.sample(range(n), max(1, round(fraction * n))))


def check_settings(trees, learning_rate, max_depth, min_leaf, subsample):
    if trees < 1:
        raise ValueError(f"trees is {trees}, but it must be at least 1")
    if not 0 < learning_rate <= 1:
        raise ValueError(f"learning_rate is {learning_rate}, but it must be above 0 and at most 1")
    if max_depth is None or max_depth < 1:
        raise ValueError(f"max_depth is {max_depth}, but boosting needs a depth limit of at least 1")
    if min_leaf < 1:
        raise ValueError(f"min_leaf is {min_leaf}, but it must be at least 1")
    if not 0 < subsample <= 1:
        raise ValueError(f"subsample is {subsample}, but it must be above 0 and at most 1")


# Gradient tree boosting  [ESL, Algorithm 10.3]
# 1. F_0(x) = the constant with the lowest loss: the mean of the targets
# 2. for m = 1..M:
#    a. r_i = y_i - F_{m-1}(x_i), the residuals
#    b. fit a regression tree to the r_i
#    c. each leaf predicts the mean of its residuals, the step with the lowest squared error
#       in that leaf (build_tree already does this)
#    d. F_m(x) = F_{m-1}(x) + learning_rate * tree_m(x)
# 3. the model is F_M
def build_boosting(rows, targets, trees, learning_rate, max_depth, min_leaf, subsample, seed):
    check_settings(trees, learning_rate, max_depth, min_leaf, subsample)
    rng = random.Random(seed)
    start = sum(targets) / len(targets)
    fitted = [start] * len(rows)
    pool = list(range(len(rows[0])))
    ans = []
    for _ in range(trees):
        r = residuals(targets, fitted)
        drawn = draw_runs(len(rows), subsample, rng)
        tree = build_tree([rows[i] for i in drawn], [r[i] for i in drawn], pool, len(pool), rng,
                          max_depth, min_leaf)
        # Shrinkage  [ESL, Section 10.12.1]
        # - each tree only takes a small step toward the residuals
        # - a smaller learning rate needs more trees, but usually predicts better on new runs
        for i in range(len(rows)):
            fitted[i] += learning_rate * predict(tree, rows[i])
        ans.append(tree)
    return {"start": start, "learning_rate": learning_rate, "trees": ans}


def predict_boosting(model, row):
    return model["start"] + model["learning_rate"] * sum(predict(tree, row) for tree in model["trees"])


# The prediction after each number of trees, 1 to M, from one trained model.
# Used to check how the error changes as trees are added.
def staged_predictions(model, row):
    ans = []
    value = model["start"]
    for tree in model["trees"]:
        value += model["learning_rate"] * predict(tree, row)
        ans.append(value)
    return ans
