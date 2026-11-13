import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


def temperature(scores, y):
    def objective(log_t):
        x = scores / np.exp(log_t)
        return float(np.mean(logsumexp(x, axis=1) - x[np.arange(len(y)), y]))

    return float(np.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))


def calibrate(candidate, baseline, labels):
    ct, bt = temperature(candidate, labels), temperature(baseline, labels)
    p, b = softmax(candidate / ct, axis=1), softmax(baseline / bt, axis=1)
    grid = [
        dict(
            weight=w, accuracy=float(np.mean((w * p + (1 - w) * b).argmax(1) == labels))
        )
        for w in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    best = max(grid, key=lambda r: r["accuracy"])
    return dict(
        candidate_temperature=ct,
        baseline_temperature=bt,
        weight=best["weight"],
        grid=grid,
    )


def blend(candidate, baseline, recipe):
    w = recipe["weight"]
    return w * softmax(candidate / recipe["candidate_temperature"], axis=1) + (
        1 - w
    ) * softmax(baseline / recipe["baseline_temperature"], axis=1)


def compare(scores, baseline, labels, seen):
    pred = scores.argmax(1)
    base = baseline.argmax(1)
    correct = pred == labels
    delta = correct.astype(float) - (base == labels).astype(float)
    novel = ~seen[np.arange(len(labels)), labels].astype(bool)
    se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
    return dict(
        accuracy=float(correct.mean()),
        delta_pp=float(delta.mean() * 100),
        paired_95ci_pp=[
            float((delta.mean() - 1.96 * se) * 100),
            float((delta.mean() + 1.96 * se) * 100),
        ],
        wins=int((delta > 0).sum()),
        losses=int((delta < 0).sum()),
        seen_accuracy=float(correct[~novel].mean()) if (~novel).any() else None,
        new_product_accuracy=float(correct[novel].mean()) if novel.any() else None,
        n=len(labels),
    )
