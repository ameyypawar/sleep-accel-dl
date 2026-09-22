"""Measure what is actually in the raw files, and check it against the docs.

This script was written before any parser, to settle the on-disk format by
observation rather than assumption. It earned its place immediately: the
dataset documentation lists the sleep-stage vocabulary as ``{0,1,2,3,5}``, and
the files also contain ``4`` (356 epochs) and ``-1`` (438 epochs). Trusting the
documentation would have silently discarded about 10% of the Deep class.

It is kept in the repo, and now reuses :mod:`sleepaccel.data.raw_io`, so it
doubles as a dataset validation step: run it after extracting the archive and
it will report anything that has drifted from what the parsers expect.

    python scripts/probe_raw_format.py
    python scripts/probe_raw_format.py --subjects 5 --json results/probe.json
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))

import argparse
import collections
import json

import numpy as np

from sleepaccel.data.labels import RAW_TO_CLASS4, class_histogram, map_labels
from sleepaccel.data.raw_io import (
    discover_subjects,
    measure_sample_rate,
    read_acceleration,
    read_heart_rate,
    read_labels,
)
from sleepaccel.paths import DATA_RAW, find_dataset_root


def show_raw_lines(path: _Path, n: int = 3) -> None:
    """Print the literal first lines, so the delimiter is visible not inferred."""
    with open(path, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= n:
                break
            print(f"      {line.rstrip()!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DATA_RAW))
    parser.add_argument(
        "--subjects",
        type=int,
        default=0,
        help="limit how many subjects are fully parsed (0 = all)",
    )
    parser.add_argument("--json", default=None, help="write the summary here")
    args = parser.parse_args()

    root = find_dataset_root(_Path(args.raw_dir))
    subjects = discover_subjects(root)
    print(f"dataset root : {root}")
    print(f"subjects     : {len(subjects)} with all four files\n")

    if not subjects:
        raise SystemExit("no complete subjects found")

    # --- literal file contents, so delimiters are observed, not assumed ----
    first = subjects[0]
    print(f"raw lines for subject {first.subject_id}:")
    for label, path in (
        ("motion       (whitespace)", first.acceleration),
        ("labels       (whitespace)", first.labels),
        ("heart_rate   (comma)", first.heart_rate),
        ("steps        (comma)", first.steps),
    ):
        print(f"   {label}")
        show_raw_lines(path)

    # --- label vocabulary across every subject ----------------------------
    observed: collections.Counter = collections.Counter()
    epochs_per_subject: list[int] = []
    for subject in subjects:
        t_lab, stage = read_labels(subject.labels)
        observed.update(stage.tolist())
        epochs_per_subject.append(int(t_lab.size))

        spacing = np.unique(np.diff(t_lab))
        if spacing.size and not np.allclose(spacing, 30.0):
            print(f"   WARNING {subject.subject_id}: label spacing {spacing[:5]}")

    print("\nlabel codes observed across all subjects:")
    for code in sorted(observed):
        mapped = RAW_TO_CLASS4.get(code)
        note = "INVALID (excluded)" if mapped is None else f"-> class {mapped}"
        flag = "  <- not in dataset docs" if code in (-1, 4) else ""
        print(f"   {code:>3}: {observed[code]:>6} epochs  {note}{flag}")

    all_stage = np.concatenate(
        [read_labels(s.labels)[1] for s in subjects]
    )
    class4, valid = map_labels(all_stage)
    print("\n4-class distribution:", class_histogram(class4))
    print(f"usable epochs: {int(valid.sum())} of {valid.size}")

    # --- per-subject signal facts -----------------------------------------
    limit = args.subjects or len(subjects)
    print(f"\nper-subject signal summary (first {limit}):")
    header = f"{'subject':>9} {'epochs':>7} {'accel_n':>9} {'Hz':>6} {'p99_dt':>7} {'hr_n':>6} {'covers':>7}"
    print(header)
    print("-" * len(header))

    rates: list[float] = []
    rows: list[dict] = []
    for subject in subjects[:limit]:
        t_lab, _ = read_labels(subject.labels)
        window = (float(t_lab[0]), float(t_lab[-1]) + 30.0)
        t_acc, _ = read_acceleration(subject.acceleration, *window)
        t_hr, _ = read_heart_rate(subject.heart_rate, *window)

        rate = measure_sample_rate(t_acc)
        rates.append(rate)
        dt = np.diff(t_acc)
        p99 = float(np.percentile(dt[dt > 0], 99)) if dt.size else 0.0
        covers = bool(t_acc.size and t_acc.min() <= window[0] and t_acc.max() >= window[1] - 30)

        print(
            f"{subject.subject_id:>9} {t_lab.size:>7} {t_acc.size:>9} "
            f"{rate:>6.1f} {p99:>7.4f} {t_hr.size:>6} {str(covers):>7}"
        )
        rows.append(
            {
                "subject_id": subject.subject_id,
                "n_epochs": int(t_lab.size),
                "n_accel_samples": int(t_acc.size),
                "sample_rate_hz": rate,
                "p99_dt": p99,
                "n_hr_samples": int(t_hr.size),
                "covers_label_window": covers,
            }
        )

    print(
        f"\nsample rate across subjects: min={min(rates):.1f} Hz "
        f"max={max(rates):.1f} Hz -- not constant, so resampling onto a fixed "
        "grid is required rather than optional"
    )
    print(
        f"epochs per subject: min={min(epochs_per_subject)} "
        f"max={max(epochs_per_subject)} total={sum(epochs_per_subject)}"
    )

    if args.json:
        out = _Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "dataset_root": str(root),
                    "n_subjects": len(subjects),
                    "label_codes": {str(k): int(v) for k, v in sorted(observed.items())},
                    "class_distribution": class_histogram(class4),
                    "subjects": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
