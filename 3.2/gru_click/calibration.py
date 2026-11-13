"""Small regularized class offsets, fitted before the calibration window."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


def fit_bias(scores, labels, penalty):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, c = scores.shape
    if penalty <= 0 or len(labels) != n or not np.isfinite(scores).all():
        raise ValueError("Invalid bias fitting inputs")

    def objective(raw):
        bias = raw - raw.mean()
        logits = scores + bias
        value = np.mean(logsumexp(logits, axis=1) - logits[np.arange(n), labels])
        value += 0.5 * penalty * np.mean(bias**2)
        gradient = softmax(logits, axis=1)
        gradient[np.arange(n), labels] -= 1
        gradient = gradient.mean(axis=0) + penalty * bias / c
        gradient -= gradient.mean()
        return float(value), gradient

    result = minimize(
        objective,
        np.zeros(c),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-1.0, 1.0)] * c,
        options=dict(maxiter=150, ftol=1e-11),
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Bias fitting failed: {result.message}")
    return result.x - result.x.mean()
