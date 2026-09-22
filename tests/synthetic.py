"""Synthetic recordings so the whole test suite runs without the dataset.

The real data is 200+ MB behind a download and cannot be committed, but the
failure modes this project guards against -- negative timestamps, sensor
dropouts, non-contiguous label codes -- are all reproducible in a few lines.
Building them here means the correctness tests run on a clean checkout and in
CI, which is where they are actually useful.
"""

from __future__ import annotations

import numpy as np

EPOCH_SECONDS = 30.0
NOMINAL_HZ = 50.0


def make_labels(n_epochs: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Epoch start times and raw AASM stage codes.

    Deliberately includes every valid code (0, 1, 2, 3, 5) and never emits 4,
    matching the real label vocabulary.
    """
    rng = np.random.default_rng(seed)
    starts = np.arange(n_epochs, dtype=np.float64) * EPOCH_SECONDS
    codes = rng.choice(np.array([0, 1, 2, 3, 5]), size=n_epochs)
    return starts, codes


def make_acceleration(
    epoch_starts: np.ndarray,
    hz: float = NOMINAL_HZ,
    jitter: float = 0.3,
    dropout_epochs: tuple[int, ...] = (),
    lead_in_seconds: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Irregularly-sampled triaxial acceleration covering the epoch grid.

    Args:
        epoch_starts: the epoch axis to cover.
        hz: nominal sample rate before jitter.
        jitter: fraction of the nominal interval to randomise, so the output
            is genuinely non-uniform like the real recordings.
        dropout_epochs: indices whose samples are removed entirely, simulating
            the watch dropping out.
        lead_in_seconds: if positive, prepend samples at negative timestamps,
            reproducing the pre-PSG data that the floor-division guard exists
            to handle.

    Returns:
        ``(t, xyz)`` with ``t`` shape ``[M]`` and ``xyz`` shape ``[M, 3]``.
    """
    rng = np.random.default_rng(seed)
    start = float(epoch_starts[0]) - lead_in_seconds
    end = float(epoch_starts[-1]) + EPOCH_SECONDS

    n = int((end - start) * hz)
    step = 1.0 / hz
    t = start + np.cumsum(rng.uniform((1 - jitter) * step, (1 + jitter) * step, size=n))
    t = t[t < end]

    xyz = rng.normal(0.0, 0.05, size=(t.size, 3))
    xyz[:, 2] += 1.0  # gravity on one axis, as a wrist-worn device would see

    if dropout_epochs:
        keep = np.ones(t.size, dtype=bool)
        for index in dropout_epochs:
            lo = float(epoch_starts[index])
            keep &= ~((t >= lo) & (t < lo + EPOCH_SECONDS))
        t, xyz = t[keep], xyz[keep]

    return t, xyz


def make_heart_rate(
    epoch_starts: np.ndarray, hz: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse heart-rate samples, far coarser than the accelerometer.

    Real PPG heart rate arrives every few seconds at best, which is why HR
    enters the model as per-epoch aggregates rather than as a waveform.
    """
    rng = np.random.default_rng(seed)
    start = float(epoch_starts[0])
    end = float(epoch_starts[-1]) + EPOCH_SECONDS
    t = np.arange(start, end, 1.0 / hz, dtype=np.float64)
    bpm = rng.normal(60.0, 5.0, size=t.size)
    return t, bpm
