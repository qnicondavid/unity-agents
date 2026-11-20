from functools import partial

import numpy as np


def minkowski_distance(x1, x2, p):
    if p == np.inf:
        return max(abs(x1[i] - x2[i]) for i in range(len(x1)))
    return pow(sum(pow(abs(x1[i] - x2[i]), p) for i in range(len(x1))), 1 / p)


def fit_standardizer(X):
    means = X.mean(axis=0)
    sds = X.std(axis=0)
    sds[sds < 10 * np.finfo(sds.dtype).eps] = 1.0
    return means, sds


def standardize(X, means, sds):
    return (X - means) / sds


def train_knn(rows, targets, k=5, p=2, scales=None):
    X = np.array(rows, dtype=float)
    means, sds = fit_standardizer(X)
    scales = np.ones(X.shape[1]) if scales is None else np.array(scales, dtype=float)
    return {"X": standardize(X, means, sds) / scales, "y": np.array(targets, dtype=float),
            "means": means, "sds": sds, "scales": scales, "k": k, "p": p}


def neighbors(model, x):
    distances = list(map(partial(minkowski_distance, x, p=model["p"]), model["X"]))
    return np.argsort(distances)[:model["k"]]


def predict(model, rows):
    X = standardize(np.array(rows, dtype=float), model["means"], model["sds"]) / model["scales"]
    return [float(model["y"][neighbors(model, x)].mean()) for x in X]
