# The comments follow Rasmussen and Williams (2006), Gaussian Processes for Machine Learning
# (R&W), chapters 2, 4 and 5.

import math

import torch

# from smoothest to roughest
KERNELS = ("rbf", "matern52", "matern32")
MIN_NOISE = 1e-6


def fit_scaler(X):
    means = X.mean(dim=0)
    sds = X.std(dim=0, correction=0)
    sds = torch.where(sds == 0, torch.ones_like(sds), sds)
    return (means, sds)


def apply_scaler(X, means, sds):
    return (X - means) / sds


# Distance between two runs, measured in length scales  [R&W, section 4.2]
#     r^2 = sum_d (x_d - x'_d)^2 / l_d^2
# - one length scale l_d per setting (ARD), or one shared by all settings
# - a long length scale means the setting can change a lot before the result changes,
#   so a setting with a very long length scale barely matters
def squared_distances(A, B, length_scales):
    A = A / length_scales
    B = B / length_scales
    sq = (A * A).sum(dim=1).reshape(-1, 1) + (B * B).sum(dim=1).reshape(1, -1) - 2 * A @ B.T
    return sq.clamp_min(0.0)


# Kernel: how alike the results of two runs should be, from their distance  [R&W, section 4.2]
#     rbf:       k(r) = s^2 exp(-r^2 / 2)
#     matern32:  k(r) = s^2 (1 + sqrt(3) r) exp(-sqrt(3) r)
#     matern52:  k(r) = s^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)
# where s^2 = signal variance (how far the results swing around their mean)
# - all three give s^2 for two runs with the same settings and fall towards 0 as the runs get
#   further apart
# - rbf is the smoothest; the Matern kernels allow rougher changes
def kernel(sq, signal, kind):
    if kind == "rbf":
        return signal * torch.exp(-0.5 * sq)
    # the gradient of sqrt is infinite at 0, so take the root only where the distance is above 0
    positive = sq > 0
    r = torch.where(positive, torch.sqrt(torch.where(positive, sq, torch.ones_like(sq))),
                    torch.zeros_like(sq))
    if kind == "matern32":
        a = math.sqrt(3) * r
        return signal * (1 + a) * torch.exp(-a)
    if kind == "matern52":
        a = math.sqrt(5) * r
        return signal * (1 + a + a * a / 3) * torch.exp(-a)
    raise ValueError(f"kernel is {kind!r}, but it must be one of {', '.join(KERNELS)}")


def unpack(log_params):
    length_scales = torch.exp(log_params[:-2])
    signal = torch.exp(log_params[-2])
    noise = torch.exp(log_params[-1]) + MIN_NOISE
    return length_scales, signal, noise


# Log marginal likelihood: how likely the training targets are under the model  [R&W, eq. 2.30]
#     log p(y | X) = -1/2 y^T (K + n^2 I)^-1 y - 1/2 log |K + n^2 I| - N/2 log 2 pi
# where K = the kernel between every pair of training runs, n^2 = noise variance
# - first term: how well the model fits the targets
# - second term: a penalty for a model that is too flexible
# - noise: two runs with the same settings and different seeds reach 95 at different steps,
#   and the noise term is the part of the result the settings cannot explain
# Computed with a Cholesky factor L of K + n^2 I  [R&W, algorithm 2.1]
#     log |K + n^2 I| = 2 sum_i log L_ii
def negative_log_likelihood(log_params, X, y, kind):
    length_scales, signal, noise = unpack(log_params)
    K = kernel(squared_distances(X, X, length_scales), signal, kind)
    K = K + noise * torch.eye(X.shape[0], dtype=X.dtype)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y.reshape(-1, 1), L)
    fit = 0.5 * (y.reshape(1, -1) @ alpha).squeeze()
    complexity = torch.log(torch.diagonal(L)).sum()
    return fit + complexity + 0.5 * X.shape[0] * math.log(2 * math.pi)


# Learning the kernel's settings  [R&W, section 5.4.1]
# - the length scales, the signal variance and the noise variance are chosen to maximize the
#   log marginal likelihood on the training runs, so they need no cross-validation
# - they are kept as logs, so they stay positive whatever step the optimizer takes
# - torch works out the gradient, and L-BFGS takes the steps
def fit_hyperparameters(X, y, kind, ard, steps):
    d = X.shape[1] if ard else 1
    log_params = torch.zeros(d + 2, dtype=X.dtype)
    log_params[-1] = math.log(0.1)
    log_params.requires_grad_(True)
    optimizer = torch.optim.LBFGS([log_params], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = negative_log_likelihood(log_params, X, y, kind)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_params.detach()


def train_gpr(rows, targets, kind="matern52", ard=True, steps=200):
    if kind not in KERNELS:
        raise ValueError(f"kernel is {kind!r}, but it must be one of {', '.join(KERNELS)}")
    torch.set_num_threads(1)
    X = torch.tensor(rows, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.float64).reshape(-1, 1)
    x_means, x_sds = fit_scaler(X)
    y_means, y_sds = fit_scaler(y)
    X_scaled = apply_scaler(X, x_means, x_sds)
    y_scaled = apply_scaler(y, y_means, y_sds).reshape(-1)
    log_params = fit_hyperparameters(X_scaled, y_scaled, kind, ard, steps)

    length_scales, signal, noise = unpack(log_params)
    K = kernel(squared_distances(X_scaled, X_scaled, length_scales), signal, kind)
    L = torch.linalg.cholesky(K + noise * torch.eye(X.shape[0], dtype=X.dtype))
    alpha = torch.cholesky_solve(y_scaled.reshape(-1, 1), L)
    return {"kind": kind, "X": X_scaled, "L": L, "alpha": alpha, "log_params": log_params,
            "x_means": x_means, "x_sds": x_sds, "y_means": y_means, "y_sds": y_sds}


# Prediction  [R&W, algorithm 2.1]
#     mean = k*^T (K + n^2 I)^-1 y
#     var  = k(x*, x*) - k*^T (K + n^2 I)^-1 k* + n^2
# where k* = the kernel between the new run and every training run
# - the mean is a weighted sum of the training targets: runs with settings close to the new
#   run count the most
# - the variance is small near many training runs and grows far from them; adding n^2 makes
#   it the spread of a new run's result, not only of the mean
def predict(model, rows, return_std=False):
    X = apply_scaler(torch.tensor(rows, dtype=torch.float64), model["x_means"], model["x_sds"])
    length_scales, signal, noise = unpack(model["log_params"])
    K_star = kernel(squared_distances(model["X"], X, length_scales), signal, model["kind"])
    mean = (K_star.T @ model["alpha"]).reshape(-1)
    mean = mean * model["y_sds"] + model["y_means"]
    if not return_std:
        return mean.tolist()
    v = torch.linalg.solve_triangular(model["L"], K_star, upper=False)
    var = (signal - (v * v).sum(dim=0) + noise).clamp_min(0.0)
    std = torch.sqrt(var) * model["y_sds"]
    return mean.tolist(), std.tolist()


# What the model learned
# - length scales: in standard deviations of each (logged) setting
# - signal_sd and noise_sd: in the units of the target, so with a logged target a noise_sd of
#   0.2 means runs with the same settings typically land about 20% apart from what the settings
#   predict
def learned_settings(model, features):
    length_scales, signal, noise = unpack(model["log_params"])
    scales = length_scales.tolist()
    if len(scales) == 1:
        scales = scales * len(features)
    y_sd = model["y_sds"].item()
    return {"length_scales": dict(zip(features, scales)), "signal_sd": math.sqrt(signal.item()) * y_sd,
            "noise_sd": math.sqrt(noise.item()) * y_sd}
