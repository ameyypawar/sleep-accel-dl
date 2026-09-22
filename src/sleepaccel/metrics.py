"""Evaluation metrics, including the degenerate-prediction guard.

Cohen's kappa is the headline because accuracy is meaningless here: Light is
55% of epochs, so a model that predicts Light and nothing else scores 55%
accuracy and a kappa of 0. ``is_degenerate`` catches exactly that case and is
written into every result file, so a collapsed model cannot be mistaken for a
working one.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .data.labels import CLASS_NAMES, N_CLASSES, collapse_to_binary


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {"n": 0, "kappa": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "is_degenerate": True}

    labels = list(range(N_CLASSES))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.bincount(y_pred, minlength=N_CLASSES)

    binary_true, binary_pred = collapse_to_binary(y_true), collapse_to_binary(y_pred)

    return {
        "n": int(y_true.size),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {name: float(v) for name, v in zip(CLASS_NAMES, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "predicted_support": {name: int(v) for name, v in zip(CLASS_NAMES, support)},
        "binary_kappa": float(cohen_kappa_score(binary_true, binary_pred)),
        "binary_accuracy": float((binary_true == binary_pred).mean()),
        # A class never predicted, or one class swamping everything, means the
        # model collapsed onto the majority rather than learning the task.
        "is_degenerate": bool((support == 0).any() or support.max() > 0.95 * support.sum()),
    }


def aggregate(fold_metrics: list[dict]) -> dict:
    """Mean and std across folds, alongside the pooled numbers."""
    if not fold_metrics:
        return {}
    keys = ("kappa", "accuracy", "macro_f1", "binary_kappa")
    out = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    out["n_folds"] = len(fold_metrics)
    out["any_degenerate"] = any(m["is_degenerate"] for m in fold_metrics)
    return out


def bootstrap_kappa_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    subjects: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap over subjects for the kappa difference b - a.

    Resampling subjects rather than epochs respects the fact that epochs within
    a subject are not independent. Applying the *same* resample to both arms is
    what makes it paired: bootstrapping each arm separately inflates the
    interval and will usually straddle zero, causing an under-claim.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(subjects)
    deltas = np.empty(n_boot, dtype=float)
    labels = list(range(N_CLASSES))

    for i in range(n_boot):
        picked = rng.choice(unique, size=unique.size, replace=True)
        mask = np.concatenate([np.flatnonzero(subjects == s) for s in picked])
        ka = cohen_kappa_score(y_true[mask], pred_a[mask], labels=labels)
        kb = cohen_kappa_score(y_true[mask], pred_b[mask], labels=labels)
        deltas[i] = kb - ka

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
    }
