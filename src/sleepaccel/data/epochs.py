"""Turning irregular accelerometer samples into a fixed-size epoch tensor.

Three traps live in this file. Each of them produces a plausible-looking
result rather than an error, which is why each gets an explicit guard.

**Negative timestamps.** Timestamps in these recordings are seconds relative
to PSG start, so samples captured before the PSG began are negative. The
obvious ``int(t / 30)`` truncates toward zero, which maps ``t = -15`` and
``t = +15`` both onto epoch 0 -- silently folding pre-study data into the
first labelled epoch. Only floor division is correct, and
:func:`epoch_index_for` uses it.

**Interpolating across sensor dropouts.** The watch stops reporting for
minutes at a time. ``np.interp`` across such a gap yields a perfectly smooth,
near-zero-variance segment, which is precisely the signature of deep sleep.
A model trained on that learns "sensor dropout means Deep" and posts an
excellent-looking score. :func:`epoch_quality` measures coverage and the
largest gap so those epochs can be dropped rather than invented.

**The wrong resampler.** ``scipy.signal.resample`` and ``decimate`` assume a
uniform input sample rate. This data is irregular (nominally ~50 Hz but not
evenly spaced), so those functions return a time-distorted waveform with no
warning. Linear interpolation onto an explicit uniform grid is the only
correct option here, and it is what :func:`resample_epoch` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpochQuality:
    """How trustworthy one epoch's accelerometer coverage is."""

    #: Fraction of the epoch window spanned by actual samples, in [0, 1].
    coverage: float
    #: Largest interval between consecutive samples, including the gaps from
    #: the epoch start to the first sample and the last sample to the end.
    largest_gap: float
    #: Number of raw samples falling inside the window.
    n_samples: int

    def is_usable(self, min_coverage: float, max_gap_seconds: float) -> bool:
        return (
            self.n_samples >= 2
            and self.coverage >= min_coverage
            and self.largest_gap <= max_gap_seconds
        )


def epoch_index_for(t: np.ndarray, epoch_seconds: float) -> np.ndarray:
    """Map timestamps to epoch indices with correct behaviour below zero.

    Uses floor division, so ``t = -0.1`` lands in epoch ``-1`` rather than
    being truncated into epoch ``0``. Callers are expected to discard negative
    indices explicitly; silently keeping them would merge pre-recording
    samples into the first labelled epoch.
    """
    return np.floor_divide(np.asarray(t, dtype=np.float64), float(epoch_seconds)).astype(
        np.int64
    )


def build_epoch_grid(label_t: np.ndarray, epoch_seconds: float) -> np.ndarray:
    """Epoch start times, taken from the label file.

    The label file defines the epoch axis. Deriving it from the accelerometer
    instead would let a late-starting or early-stopping watch shift every
    epoch boundary against the PSG scoring, misaligning signal and label by a
    constant offset that no metric would flag.

    Raises:
        ValueError: if the label timestamps are not evenly spaced by
            ``epoch_seconds``. That would mean the file is not a simple epoch
            series and the rest of this module's assumptions do not hold, so
            it is worth failing loudly here rather than producing a subtly
            misaligned dataset.
    """
    label_t = np.asarray(label_t, dtype=np.float64)
    if label_t.size < 2:
        return label_t

    spacing = np.diff(label_t)
    if not np.allclose(spacing, epoch_seconds):
        unique = np.unique(np.round(spacing, 6))
        raise ValueError(
            f"label timestamps are not spaced by {epoch_seconds}s; "
            f"observed spacings: {unique[:10]}"
        )
    return label_t


def epoch_quality(t: np.ndarray, start: float, end: float) -> EpochQuality:
    """Measure sample coverage and the worst gap inside one epoch window.

    The edge gaps matter as much as the interior ones: samples clustered into
    the first two seconds of a 30-second epoch have a small interior gap but
    tell you nothing about the remaining 28 seconds.
    """
    t = np.asarray(t, dtype=np.float64)
    inside = t[(t >= start) & (t < end)]
    n = int(inside.size)
    if n == 0:
        return EpochQuality(coverage=0.0, largest_gap=float(end - start), n_samples=0)

    duration = float(end - start)
    span = float(inside[-1] - inside[0])
    coverage = float(np.clip(span / duration, 0.0, 1.0)) if duration > 0 else 0.0

    # Include the lead-in and lead-out gaps so partial coverage cannot hide.
    boundaries = np.concatenate(([start], inside, [end]))
    largest_gap = float(np.max(np.diff(boundaries)))
    return EpochQuality(coverage=coverage, largest_gap=largest_gap, n_samples=n)


def resample_epoch(
    t: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    target_hz: float,
) -> np.ndarray:
    """Linearly interpolate irregular samples onto a uniform grid.

    Args:
        t: sample timestamps, shape ``[M]``, assumed sorted ascending.
        values: sample values, shape ``[M, C]``.
        start: epoch start time, inclusive.
        end: epoch end time, exclusive.
        target_hz: output sample rate.

    Returns:
        ``[C, T]`` float32, where ``T = round((end - start) * target_hz)``.

    Note that ``np.interp`` clamps outside the input range, holding the first
    and last values flat. That is acceptable only because callers gate on
    :func:`epoch_quality` first; without that gate, a dropout becomes a flat
    line that reads as deep sleep.
    """
    t = np.asarray(t, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]

    n_out = int(round((end - start) * target_hz))
    grid = start + np.arange(n_out, dtype=np.float64) / target_hz

    if t.size == 0:
        return np.zeros((values.shape[1], n_out), dtype=np.float32)
    if t.size == 1:
        return np.repeat(
            values[0].astype(np.float32)[:, None], n_out, axis=1
        )

    out = np.empty((values.shape[1], n_out), dtype=np.float32)
    for channel in range(values.shape[1]):
        out[channel] = np.interp(grid, t, values[:, channel]).astype(np.float32)
    return out


def derive_magnitude(xyz: np.ndarray) -> np.ndarray:
    """Vector magnitude channel, ``sqrt(x^2 + y^2 + z^2)``.

    Added as a fourth channel because magnitude is invariant to how the watch
    happens to be rotated on the wrist, while the individual axes are not.
    The network can learn this itself, but handing it over costs nothing and
    removes a nuisance factor from a 31-subject dataset.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    axis = 0 if xyz.shape[0] == 3 else -1
    return np.sqrt((xyz**2).sum(axis=axis)).astype(np.float32)


def epochs_from_recording(
    t: np.ndarray,
    xyz: np.ndarray,
    epoch_starts: np.ndarray,
    epoch_seconds: float,
    target_hz: float,
    min_coverage: float,
    max_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray, list[EpochQuality]]:
    """Build the per-epoch waveform tensor for one subject.

    Returns:
        ``(waveform, keep, quality)`` where ``waveform`` is
        ``[N_epochs, 4, T]`` (x, y, z, magnitude), ``keep`` is a boolean mask
        of epochs that passed the coverage and gap thresholds, and ``quality``
        holds the per-epoch measurements for the cache manifest.

    Epochs that fail the quality gate are still emitted, as zeros, so the
    array stays aligned with the label axis. The ``keep`` mask is what excludes
    them downstream; dropping rows here would desynchronise signal and labels,
    which is exactly the failure this module exists to prevent.
    """
    t = np.asarray(t, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    n_epochs = int(len(epoch_starts))
    n_out = int(round(epoch_seconds * target_hz))

    waveform = np.zeros((n_epochs, 4, n_out), dtype=np.float32)
    keep = np.zeros(n_epochs, dtype=bool)
    quality: list[EpochQuality] = []

    # One pass with searchsorted rather than a boolean scan per epoch: the
    # recording holds millions of samples and an O(N_epochs * M) scan is the
    # difference between seconds and minutes per subject.
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    xyz_sorted = xyz[order]

    for i, start in enumerate(np.asarray(epoch_starts, dtype=np.float64)):
        end = start + epoch_seconds
        lo, hi = np.searchsorted(t_sorted, (start, end))
        window_t = t_sorted[lo:hi]
        window_xyz = xyz_sorted[lo:hi]

        q = epoch_quality(window_t, start, end)
        quality.append(q)
        if not q.is_usable(min_coverage, max_gap_seconds):
            continue

        resampled = resample_epoch(window_t, window_xyz, start, end, target_hz)
        waveform[i, :3] = resampled
        waveform[i, 3] = derive_magnitude(resampled)
        keep[i] = True

    return waveform, keep, quality
