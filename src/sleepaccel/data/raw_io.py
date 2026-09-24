"""Parsers for the raw PhysioNet text files.

Written against measured file contents, not the dataset documentation, because
the two disagree in ways that matter. What was actually observed across all 31
subjects:

``motion/<id>_acceleration.txt``
    Whitespace-delimited, four columns: ``t x y z``. Acceleration in g.
    Roughly 1.1-2.0 million rows per subject.

``heart_rate/<id>_heartrate.txt``
    **Comma**-delimited, two columns: ``t bpm``.

``steps/<id>_steps.txt``
    **Comma**-delimited, two columns: ``t steps``. Unused -- see below.

``labels/<id>_labeled_sleep.txt``
    Whitespace-delimited, two columns: ``t stage``. Always starts at t=0 and
    steps by exactly 30s, on every subject.

The delimiter is not consistent across file types. A single shared parser that
assumed one of them would fail on the others, and `np.loadtxt` defaults to
whitespace, so the comma-delimited files would parse as a single garbled
column rather than raising anything obvious.

Timestamps are seconds relative to PSG start and run far into the negative:
the accelerometer files begin between roughly -1,800 and -486,000 seconds,
the latter being 5.6 days of pre-study wear. Heart rate reaches -355,000.
Those samples are real but irrelevant, and they are also the bulk of the data
for some subjects, so they are trimmed against the label window on load.

Sampling rate is nominally 50 Hz but genuinely varies: the median interval is
0.0200s for most subjects and 0.0150s (66.6 Hz) for subject 3509524. Anything
assuming a fixed rate across subjects is wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..paths import SubjectFiles

#: Directory names inside the extracted PhysioNet archive.
MOTION_DIR = "motion"
LABELS_DIR = "labels"
HEART_RATE_DIR = "heart_rate"
STEPS_DIR = "steps"

#: Filename suffixes, which differ per directory.
MOTION_SUFFIX = "_acceleration.txt"
LABELS_SUFFIX = "_labeled_sleep.txt"
HEART_RATE_SUFFIX = "_heartrate.txt"
STEPS_SUFFIX = "_steps.txt"

#: Padding kept either side of the label window when trimming. One epoch is
#: enough for the resampler's edge interpolation and avoids discarding samples
#: that legitimately belong to the first and last epochs.
TRIM_PAD_SECONDS = 30.0


def discover_subjects(dataset_root: Path) -> list[SubjectFiles]:
    """Find every subject that has all four files.

    Subject ids are numeric strings of differing length (``1066528``,
    ``46343``). They are sorted as integers so the ordering is stable and
    intuitive; since fold assignment is seeded off this list, the sort order
    is part of the experiment's reproducibility and is therefore pinned here
    rather than left to whatever the filesystem returns.
    """
    dataset_root = Path(dataset_root)
    labels_dir = dataset_root / LABELS_DIR

    subjects: list[SubjectFiles] = []
    for path in sorted(labels_dir.glob(f"*{LABELS_SUFFIX}")):
        subject_id = path.name[: -len(LABELS_SUFFIX)]
        files = SubjectFiles(
            subject_id=subject_id,
            acceleration=dataset_root / MOTION_DIR / f"{subject_id}{MOTION_SUFFIX}",
            heart_rate=dataset_root / HEART_RATE_DIR / f"{subject_id}{HEART_RATE_SUFFIX}",
            steps=dataset_root / STEPS_DIR / f"{subject_id}{STEPS_SUFFIX}",
            labels=path,
        )
        if all(
            p.exists()
            for p in (files.acceleration, files.heart_rate, files.steps, files.labels)
        ):
            subjects.append(files)

    return sorted(subjects, key=lambda s: int(s.subject_id))


def _ensure_sorted(t: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sort by timestamp if needed.

    Every downstream function uses ``np.searchsorted`` on these timestamps,
    which returns silently wrong windows on unsorted input rather than
    raising.
    """
    if np.all(np.diff(t) >= 0):
        return (t, *arrays)
    order = np.argsort(t, kind="stable")
    return (t[order], *(a[order] for a in arrays))


def read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``t stage`` pairs. Returns ``(epoch_start_seconds, raw_stage)``."""
    raw = np.loadtxt(path, dtype=np.float64)
    raw = np.atleast_2d(raw)
    t, stage = _ensure_sorted(raw[:, 0], raw[:, 1].astype(np.int64))
    return t, stage


def read_acceleration(
    path: Path, t_min: float | None = None, t_max: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read whitespace-delimited ``t x y z``. Returns ``(t, xyz)``.

    ``t_min``/``t_max`` trim to the study window before anything else touches
    the array. For the subject whose file starts 5.6 days early this discards
    the majority of 1.8 million rows, which is the difference between a cache
    build that takes seconds and one that takes minutes.
    """
    raw = np.loadtxt(path, dtype=np.float64)
    raw = np.atleast_2d(raw)
    if raw.shape[1] != 4:
        raise ValueError(
            f"{path}: expected 4 whitespace-delimited columns (t x y z), "
            f"got {raw.shape[1]}"
        )

    t, xyz = raw[:, 0], raw[:, 1:4]
    if t_min is not None:
        keep = t >= t_min - TRIM_PAD_SECONDS
        t, xyz = t[keep], xyz[keep]
    if t_max is not None:
        keep = t <= t_max + TRIM_PAD_SECONDS
        t, xyz = t[keep], xyz[keep]

    t, xyz = _ensure_sorted(t, xyz)
    return t, xyz


def read_heart_rate(
    path: Path, t_min: float | None = None, t_max: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read comma-delimited ``t,bpm``. Returns ``(t, bpm)``.

    Note the delimiter: this file is comma-separated while the motion and
    label files are whitespace-separated.
    """
    raw = np.loadtxt(path, dtype=np.float64, delimiter=",")
    raw = np.atleast_2d(raw)
    if raw.shape[1] != 2:
        raise ValueError(
            f"{path}: expected 2 comma-delimited columns (t,bpm), got {raw.shape[1]}"
        )

    t, bpm = raw[:, 0], raw[:, 1]
    if t_min is not None:
        keep = t >= t_min - TRIM_PAD_SECONDS
        t, bpm = t[keep], bpm[keep]
    if t_max is not None:
        keep = t <= t_max + TRIM_PAD_SECONDS
        t, bpm = t[keep], bpm[keep]

    t, bpm = _ensure_sorted(t, bpm)
    return t, bpm


def read_steps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read comma-delimited ``t,steps``.

    Provided for completeness and deliberately unused. Step counts are a
    derived product of the same accelerometer this project is testing, so
    feeding them to the model would smuggle processed motion information into
    an experiment whose entire point is to measure what raw motion alone can
    do.
    """
    raw = np.atleast_2d(np.loadtxt(path, dtype=np.float64, delimiter=","))
    t, steps = _ensure_sorted(raw[:, 0], raw[:, 1])
    return t, steps


def measure_sample_rate(t: np.ndarray) -> float:
    """Median sampling rate in Hz, for the cache manifest.

    Reported per subject rather than assumed, because it is not constant
    across the cohort.
    """
    if t.size < 2:
        return 0.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))
