"""Torch dataset over the cached epochs, windowed into sequences.

The model classifies an epoch in the context of its neighbours, because Deep
and REM are close to indistinguishable from motion in a single isolated
30-second window -- both are near-motionless -- but separable once you can see
where they sit in the night's cycle. So the unit of training is a window of
``context_len`` consecutive epochs, not one epoch.

Two properties of the windowing are load-bearing:

**Windows never span a subject boundary.** The naive implementation
concatenates every subject's epochs into one long array and slides over it,
which silently produces windows holding the end of one person's night and the
start of another's. It does not crash and barely moves the loss. It just
invalidates the protocol.

**Evaluation tiles with almost no overlap.** Training uses a short stride so
windows overlap, which is free augmentation. Evaluation uses a stride equal to
the window length, so interior epochs are predicted once -- but the last window
of each recording is pulled back to finish on the final epoch, and so repeats
up to ``context_len - 1`` epochs. That is 1.1% of evaluated epochs on this
dataset; see :mod:`sleepaccel.data.windows` for the measurement and for
:func:`~sleepaccel.data.windows.evaluation_epoch_order`, which exposes the
repeats so they can be removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .hr_features import N_HR_FEATURES
from .labels import INVALID, N_CLASSES
from .windows import window_index


@dataclass(frozen=True)
class Window:
    """One training sample: a run of consecutive epochs from one subject."""

    subject_id: str
    start: int


class SubjectCache:
    """Lazy reader for one subject's cached ``.npz``.

    Loaded on first access and held, because a subject's arrays are a few tens
    of megabytes and the sampler revisits the same subject many times within an
    epoch.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, np.ndarray] | None = None

    @property
    def data(self) -> dict[str, np.ndarray]:
        if self._data is None:
            with np.load(self.path) as handle:
                self._data = {k: handle[k] for k in handle.files}
        return self._data

    @property
    def n_epochs(self) -> int:
        return int(self.data["labels"].size)


class SleepWindows(Dataset):
    """Windows of consecutive epochs, drawn from a fixed set of subjects."""

    def __init__(
        self,
        subject_ids: list[str],
        cache_dir: str | Path,
        context_len: int = 20,
        stride: int = 5,
        input_variant: str = "accel_only",
        normalizer: tuple[np.ndarray, np.ndarray] | None = None,
        hr_normalizer: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.context_len = context_len
        self.stride = stride
        self.input_variant = input_variant
        self.normalizer = normalizer
        self.hr_normalizer = hr_normalizer

        self.caches: dict[str, SubjectCache] = {
            sid: SubjectCache(self.cache_dir / f"{sid}.npz") for sid in subject_ids
        }

        # Built per subject, so a window can never straddle two people.
        self.windows: list[Window] = []
        for sid in subject_ids:
            for start in window_index(
                self.caches[sid].n_epochs, context_len, stride
            ):
                self.windows.append(Window(sid, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        data = self.caches[window.subject_id].data
        lo, hi = window.start, window.start + self.context_len

        waveform = data["waveform"][lo:hi].astype(np.float32)
        labels = data["labels"][lo:hi].astype(np.int64)
        keep = data["keep"][lo:hi]
        hr_feats = data["hr_feats"][lo:hi].astype(np.float32)
        hr_valid = data["hr_valid"][lo:hi].astype(np.float32)

        if self.normalizer is not None:
            mean, std = self.normalizer
            waveform = (waveform - mean[None, :, None]) / std[None, :, None]
        if self.hr_normalizer is not None:
            mean, std = self.hr_normalizer
            hr_feats = (hr_feats - mean[None, :]) / std[None, :]

        # Epochs that failed the quality gate or lack a label are masked out of
        # the loss rather than removed, so the sequence stays contiguous in
        # time and the context model sees the real gap structure.
        labels = np.where(keep, labels, INVALID)

        return {
            "waveform": torch.from_numpy(np.ascontiguousarray(waveform)),
            "hr": torch.from_numpy(
                np.concatenate([hr_feats, hr_valid[:, None]], axis=1)
            ),
            "labels": torch.from_numpy(labels),
            "valid": torch.from_numpy(keep.astype(np.bool_)),
        }


def fit_normalizer(
    subject_ids: list[str], cache_dir: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel waveform mean and std, over training subjects only.

    Fitting this across every subject would leak held-out information into all
    folds at once. The effect is small and uniform, which makes it look like a
    genuine improvement rather than a bug.
    """
    cache_dir = Path(cache_dir)
    total = np.zeros(4, dtype=np.float64)
    total_sq = np.zeros(4, dtype=np.float64)
    count = 0

    for sid in subject_ids:
        with np.load(cache_dir / f"{sid}.npz") as handle:
            waveform = handle["waveform"][handle["keep"]]
        if waveform.size == 0:
            continue
        flat = waveform.transpose(1, 0, 2).reshape(4, -1)
        total += flat.sum(axis=1)
        total_sq += (flat**2).sum(axis=1)
        count += flat.shape[1]

    if count == 0:
        return np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)

    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def fit_hr_normalizer(
    subject_ids: list[str], cache_dir: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the HR features, over training subjects only."""
    cache_dir = Path(cache_dir)
    chunks: list[np.ndarray] = []
    for sid in subject_ids:
        with np.load(cache_dir / f"{sid}.npz") as handle:
            feats = handle["hr_feats"][handle["hr_valid"]]
        if feats.size:
            chunks.append(feats)

    if not chunks:
        return (
            np.zeros(N_HR_FEATURES, dtype=np.float32),
            np.ones(N_HR_FEATURES, dtype=np.float32),
        )

    stacked = np.concatenate(chunks, axis=0)
    std = stacked.std(axis=0)
    return stacked.mean(axis=0).astype(np.float32), np.where(std > 0, std, 1.0).astype(
        np.float32
    )


def class_counts(subject_ids: list[str], cache_dir: str | Path) -> np.ndarray:
    """Per-class epoch counts over training subjects, for loss weighting."""
    cache_dir = Path(cache_dir)
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for sid in subject_ids:
        with np.load(cache_dir / f"{sid}.npz") as handle:
            labels = handle["labels"][handle["keep"]]
        for c in range(N_CLASSES):
            counts[c] += int((labels == c).sum())
    return counts


def make_loader(
    dataset: SleepWindows, batch_size: int, shuffle: bool, workers: int = 0
) -> DataLoader:
    """DataLoader with workers off by default.

    On macOS, DataLoader workers use spawn, so each re-imports torch from
    scratch. With a cache this small the workers cost more to start than the
    loading they save.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
    )
