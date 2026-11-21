from __future__ import annotations

import numpy as np


def regression_report(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.maximum(np.asarray(y_pred, float), 0)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "RMSLE": float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))),
        "within_10pct": float(np.mean(np.abs(err) <= 0.10 * y_true)),
        "within_20pct": float(np.mean(np.abs(err) <= 0.20 * y_true)),
    }


def interval_report(y_true, low, high) -> dict:
    y_true = np.asarray(y_true, float)
    return {"coverage": float(np.mean((y_true >= low) & (y_true <= high))),
            "median_rel_width": float(np.median((high - low) / y_true))}


def bootstrap_ci(y_true, y_pred, stat=lambda t, p: np.mean(np.abs(p - t)), n=500, seed=0):
    """95% bootstrap CI of a paired statistic (default MAE)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = [stat(y_true[i], y_pred[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
