"""One-shot probe of the raw sleep-accel dataset's on-disk format.

Why this exists: before writing a parser, we need ground truth about the
actual file format -- delimiter, column count, timestamp range, the exact
set of label values that occur, and the empirical accelerometer sampling
interval -- rather than assuming the PhysioNet documentation matches the
delivered files exactly. Run this once, by hand, after the archive is
extracted and before any parsing code in sleepaccel is written or trusted.

This script is read-only: it prints results and writes nothing to disk. It
must fail with DatasetNotFoundError (not a stack trace from deeper in the
code) when the dataset has not been extracted yet, since that is the
expected state for most of Phase 0.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Allow running directly (`.venv/bin/python scripts/probe_raw_format.py`)
# without requiring PYTHONPATH=src to be set.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sleepaccel.paths import DATA_RAW, SubjectFiles, find_dataset_root  # noqa: E402

# Sibling directory names and per-subject filename suffixes within the
# dataset root, per the PhysioNet "sleep-accel" layout. `labeled_sleep` is
# also the structural marker find_dataset_root() looks for.
_FILE_TYPE_DIRS = {
    "acceleration": "motion",
    "heart_rate": "heart_rate",
    "steps": "steps",
    "labels": "labeled_sleep",
}
_FILE_SUFFIXES = {
    "acceleration": "_acceleration.txt",
    "heart_rate": "_heartrate.txt",
    "steps": "_steps.txt",
    "labels": "_labeled_sleep.txt",
}


def _detect_delimiter(sample_line: str) -> str:
    if "," in sample_line:
        return ","
    if "\t" in sample_line:
        return "\\t"
    return "whitespace"


def _split(line: str, delimiter: str) -> list[str]:
    if delimiter == ",":
        return [tok.strip() for tok in line.split(",")]
    if delimiter == "\\t":
        return [tok.strip() for tok in line.split("\t")]
    return line.split()


def _read_nonempty_lines(path: Path) -> list[str]:
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _probe_file(path: Path, label: str) -> list[str]:
    """Print probe stats for one file. Returns its non-empty lines."""
    print(f"  [{label}] {path}")
    if not path.is_file():
        print("    MISSING")
        return []
    lines = _read_nonempty_lines(path)
    if not lines:
        print("    EMPTY FILE")
        return []

    delim = _detect_delimiter(lines[0])
    ncols = len(_split(lines[0], delim))
    print(f"    delimiter={delim!r} columns={ncols} rows={len(lines)}")

    print("    first 5 lines:")
    for ln in lines[:5]:
        print(f"      {ln}")
    print("    last 5 lines:")
    for ln in lines[-5:]:
        print(f"      {ln}")

    timestamps: list[float] = []
    for ln in lines:
        tok = _split(ln, delim)[0]
        try:
            timestamps.append(float(tok))
        except ValueError:
            pass
    if timestamps:
        print(f"    timestamp range: min={min(timestamps):.3f} max={max(timestamps):.3f}")
    else:
        print("    no parseable numeric first column")
    return lines


def main() -> None:
    dataset_root = find_dataset_root(DATA_RAW)
    print(f"dataset root: {dataset_root}\n")

    type_dirs = {ft: dataset_root / d for ft, d in _FILE_TYPE_DIRS.items()}
    for ft, d in type_dirs.items():
        if not d.is_dir():
            print(f"warning: expected directory for {ft!r} not found: {d}")

    subject_ids_per_type: dict[str, set[str]] = {}
    for ft, d in type_dirs.items():
        suffix = _FILE_SUFFIXES[ft]
        subject_ids_per_type[ft] = (
            {p.name[: -len(suffix)] for p in d.glob(f"*{suffix}")} if d.is_dir() else set()
        )

    all_subject_ids = sorted(set().union(*subject_ids_per_type.values()))
    print(f"total subjects found: {len(all_subject_ids)}\n")

    print("subjects missing one or more of the four files:")
    any_missing = False
    for sid in all_subject_ids:
        missing = [ft for ft, ids in subject_ids_per_type.items() if sid not in ids]
        if missing:
            any_missing = True
            print(f"  {sid}: missing {missing}")
    if not any_missing:
        print("  none")
    print()

    probe_subject_ids = all_subject_ids[:2]
    accel_timestamp_lists: list[list[float]] = []

    for sid in probe_subject_ids:
        print(f"=== subject {sid} ===")
        files = SubjectFiles(
            subject_id=sid,
            acceleration=type_dirs["acceleration"] / f"{sid}{_FILE_SUFFIXES['acceleration']}",
            heart_rate=type_dirs["heart_rate"] / f"{sid}{_FILE_SUFFIXES['heart_rate']}",
            steps=type_dirs["steps"] / f"{sid}{_FILE_SUFFIXES['steps']}",
            labels=type_dirs["labels"] / f"{sid}{_FILE_SUFFIXES['labels']}",
        )
        accel_lines = _probe_file(files.acceleration, "acceleration")
        _probe_file(files.heart_rate, "heart_rate")
        _probe_file(files.steps, "steps")
        _probe_file(files.labels, "labels")

        ts: list[float] = []
        for ln in accel_lines:
            delim = _detect_delimiter(ln)
            tok = _split(ln, delim)[0]
            try:
                ts.append(float(tok))
            except ValueError:
                pass
        ts.sort()
        accel_timestamp_lists.append(ts)
        print()

    # Distinct label values across ALL subjects, not just the two probed
    # above -- a rare label value that only appears in subject #30 would
    # otherwise be missed, and the whole point of this probe is to catch
    # that before it becomes a silent bug in the label encoder.
    print("distinct label values across all subjects:")
    label_values: Counter[str] = Counter()
    for sid in all_subject_ids:
        label_path = type_dirs["labels"] / f"{sid}{_FILE_SUFFIXES['labels']}"
        if not label_path.is_file():
            continue
        for ln in _read_nonempty_lines(label_path):
            delim = _detect_delimiter(ln)
            tok = _split(ln, delim)[-1]
            label_values[tok] += 1
    for value, count in sorted(label_values.items()):
        print(f"  {value!r}: {count} epochs")
    print()

    # Empirical accelerometer sampling interval, from the two probed
    # subjects. Deltas are computed per-subject and only then concatenated,
    # so the (meaningless) gap between two different subjects' timestamp
    # references never pollutes the distribution.
    print("accelerometer sampling interval (from probed subjects):")
    delta_arrays = [np.diff(np.array(ts)) for ts in accel_timestamp_lists if len(ts) > 1]
    deltas = np.concatenate(delta_arrays) if delta_arrays else np.array([])
    deltas = deltas[deltas > 0]
    if len(deltas) > 0:
        median = float(np.median(deltas))
        p1 = float(np.percentile(deltas, 1))
        p99 = float(np.percentile(deltas, 99))
        print(f"  median delta-t: {median:.6f} s (implied {1.0 / median:.2f} Hz)")
        print(f"  1st pct delta-t: {p1:.6f} s (implied {1.0 / p1:.2f} Hz)")
        print(f"  99th pct delta-t: {p99:.6f} s (implied {1.0 / p99:.2f} Hz)")
    else:
        print("  not enough samples to compute delta-t distribution")


if __name__ == "__main__":
    main()
