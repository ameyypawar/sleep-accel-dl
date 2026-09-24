"""Per-epoch heart-rate aggregates for the ablation arm.

Heart rate cannot enter the model the way acceleration does. PPG heart rate
arrives every few seconds at best, against roughly 50 Hz for the
accelerometer -- three orders of magnitude sparser. Resampling it onto the
same 900-sample grid would manufacture a waveform out of a handful of real
readings, and the network would spend capacity modelling interpolation
artefacts. So HR is summarised per epoch instead, and concatenated to the
epoch embedding after the convolutional encoder.

The missing-data policy matters here. An epoch with no HR reading gets zeroed
features *and* a ``hr_valid`` flag, and that flag is handed to the model as an
extra input. Without it, "no reading" and "0 bpm" are the same vector, and 0
bpm is not a neutral value -- it is a physiologically extreme one that the
network would learn to treat as signal.
"""

from __future__ import annotations

import numpy as np

#: mean, std, min, max, and deviation from the subject's own baseline.
N_HR_FEATURES = 5
HR_FEATURE_NAMES: tuple[str, ...] = (
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_delta_baseline",
)


def subject_hr_baseline(hr_bpm: np.ndarray) -> float:
    """Median heart rate across the whole recording.

    Resting heart rate varies enormously between people, so an absolute bpm
    tells the model less than a deviation from that person's own norm. The
    median is used rather than the mean because it is unmoved by the brief
    tachycardic spikes around awakenings.

    This is deliberately transductive: it is computed over the entire night,
    including epochs that will later land in a held-out fold. That is a
    considered trade-off rather than an oversight. The alternative -- a
    baseline from the training subjects only -- would be meaningless, since
    the quantity is per-subject by construction, and at inference time on a
    real device the whole night is available before scoring anyway.
    """
    hr_bpm = np.asarray(hr_bpm, dtype=np.float64)
    finite = hr_bpm[np.isfinite(hr_bpm)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def epoch_hr_features(
    hr_t: np.ndarray,
    hr_bpm: np.ndarray,
    epoch_starts: np.ndarray,
    epoch_seconds: float,
    baseline: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarise heart rate within each epoch window.

    Args:
        hr_t: heart-rate sample timestamps, shape ``[M]``.
        hr_bpm: heart-rate values in bpm, shape ``[M]``.
        epoch_starts: epoch start times, shape ``[N]``.
        epoch_seconds: epoch length.
        baseline: subject baseline; computed with
            :func:`subject_hr_baseline` when omitted.

    Returns:
        ``(features, valid)`` with ``features`` shape ``[N, 5]`` float32 and
        ``valid`` a boolean mask, False where the epoch contained no reading.
        Invalid rows are all-zero.
    """
    hr_t = np.asarray(hr_t, dtype=np.float64)
    hr_bpm = np.asarray(hr_bpm, dtype=np.float64)
    epoch_starts = np.asarray(epoch_starts, dtype=np.float64)

    if baseline is None:
        baseline = subject_hr_baseline(hr_bpm)

    n_epochs = int(epoch_starts.size)
    features = np.zeros((n_epochs, N_HR_FEATURES), dtype=np.float32)
    valid = np.zeros(n_epochs, dtype=bool)

    if hr_t.size == 0:
        return features, valid

    order = np.argsort(hr_t, kind="stable")
    t_sorted = hr_t[order]
    bpm_sorted = hr_bpm[order]

    for i, start in enumerate(epoch_starts):
        lo, hi = np.searchsorted(t_sorted, (start, start + epoch_seconds))
        window = bpm_sorted[lo:hi]
        window = window[np.isfinite(window)]
        if window.size == 0:
            continue

        features[i] = (
            window.mean(),
            # A single reading has no spread; report 0 rather than NaN.
            window.std() if window.size > 1 else 0.0,
            window.min(),
            window.max(),
            window.mean() - baseline,
        )
        valid[i] = True

    return features, valid


def normalize_hr_features(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Standardise HR features using statistics fitted on training subjects.

    Kept separate from :func:`epoch_hr_features` so the statistics can be
    computed per fold. Fitting them across all subjects would leak held-out
    information into every fold uniformly, which inflates results in a way
    that looks like a genuine improvement.
    """
    std = np.where(std > 0, std, 1.0)
    return ((np.asarray(features, dtype=np.float32) - mean) / std).astype(np.float32)
