"""Precompute the per-epoch tensors once and cache them to disk.

Re-parsing 1-2 million rows of text per subject on every training epoch would
dominate the run, and the parse result never changes. So the whole pipeline --
trim, align to the label grid, resample, gate on quality, aggregate heart rate
-- runs once and lands in an ``.npz`` per subject.

The cache is keyed by a hash of only the config fields that actually change
its contents (epoch length, resample rate, coverage and gap thresholds). A
learning-rate change leaves it valid, which is what makes iterating on the
model cheap.

Alongside the arrays it writes ``manifest.json``: per-subject epoch counts,
how many epochs were dropped and why, the measured sample rate, and the class
histogram. That file is the evidence the build worked. Two things to check in
it, because both indicate silent misalignment rather than a crash:

* a roughly uniform class histogram -- real nights are dominated by Light
* a dropped-epoch rate in the tens of percent rather than single digits
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig, config_hash, load_config
from ..paths import DATA_RAW, SubjectFiles, cache_dir, find_dataset_root
from .epochs import epochs_from_recording
from .hr_features import epoch_hr_features, subject_hr_baseline
from .labels import class_histogram, map_labels
from .raw_io import (
    discover_subjects,
    measure_sample_rate,
    read_acceleration,
    read_heart_rate,
    read_labels,
)


@dataclass
class SubjectCacheResult:
    """What one subject contributed, for the manifest."""

    subject_id: str
    n_epochs: int
    n_kept: int
    n_label_invalid: int
    n_signal_rejected: int
    sample_rate_hz: float
    hr_valid_fraction: float
    class_counts: dict[str, int]


def build_subject_cache(
    files: SubjectFiles, cfg: ExperimentConfig, out_dir: Path
) -> SubjectCacheResult:
    """Build and write one subject's epoch cache."""
    epoch_starts, raw_stage = read_labels(files.labels)
    window = (float(epoch_starts[0]), float(epoch_starts[-1]) + cfg.epoch_seconds)

    t_acc, xyz = read_acceleration(files.acceleration, *window)
    t_hr, bpm = read_heart_rate(files.heart_rate, *window)

    waveform, signal_ok, quality = epochs_from_recording(
        t_acc,
        xyz,
        epoch_starts,
        cfg.epoch_seconds,
        cfg.resample_hz,
        cfg.min_epoch_coverage,
        cfg.max_gap_seconds,
    )

    labels, label_ok = map_labels(raw_stage)
    hr_feats, hr_valid = epoch_hr_features(
        t_hr, bpm, epoch_starts, cfg.epoch_seconds, subject_hr_baseline(bpm)
    )

    # An epoch is usable only if both the signal passed its quality gate and a
    # human actually scored it. Keeping either half alone would mean training
    # on invented signal or on absent labels.
    keep = signal_ok & label_ok

    np.savez_compressed(
        out_dir / f"{files.subject_id}.npz",
        waveform=waveform.astype(np.float32),
        hr_feats=hr_feats.astype(np.float32),
        hr_valid=hr_valid,
        labels=labels.astype(np.int64),
        keep=keep,
        epoch_start_s=np.asarray(epoch_starts, dtype=np.float64),
        coverage=np.array([q.coverage for q in quality], dtype=np.float32),
        largest_gap=np.array([q.largest_gap for q in quality], dtype=np.float32),
    )

    return SubjectCacheResult(
        subject_id=files.subject_id,
        n_epochs=int(labels.size),
        n_kept=int(keep.sum()),
        n_label_invalid=int((~label_ok).sum()),
        n_signal_rejected=int((~signal_ok).sum()),
        sample_rate_hz=measure_sample_rate(t_acc),
        hr_valid_fraction=float(hr_valid.mean()) if hr_valid.size else 0.0,
        class_counts=class_histogram(np.where(keep, labels, -1)),
    )


def _worker(payload: tuple[SubjectFiles, ExperimentConfig, Path]) -> SubjectCacheResult:
    return build_subject_cache(*payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw-dir", default=str(DATA_RAW))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    digest = config_hash(cfg)
    out_dir = cache_dir(digest)

    subjects = discover_subjects(find_dataset_root(Path(args.raw_dir)))
    if not args.force:
        subjects = [s for s in subjects if not (out_dir / f"{s.subject_id}.npz").exists()]

    print(f"cache dir : {out_dir}")
    print(f"building  : {len(subjects)} subjects "
          f"(epoch={cfg.epoch_seconds}s, resample={cfg.resample_hz}Hz)")

    results: list[SubjectCacheResult] = []
    if subjects:
        payloads = [(s, cfg, out_dir) for s in subjects]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, result in enumerate(pool.map(_worker, payloads), start=1):
                results.append(result)
                print(
                    f"  [{n}/{len(subjects)}] {result.subject_id}: "
                    f"{result.n_kept}/{result.n_epochs} epochs kept "
                    f"({result.sample_rate_hz:.1f} Hz)",
                    flush=True,
                )

    manifest_path = out_dir / "manifest.json"
    existing: dict = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))

    per_subject = {r.subject_id: r.__dict__ for r in results}
    per_subject = {**existing.get("subjects", {}), **per_subject}

    totals = {"Wake": 0, "Light": 0, "Deep": 0, "REM": 0, "Invalid": 0}
    for entry in per_subject.values():
        for name, count in entry["class_counts"].items():
            totals[name] = totals.get(name, 0) + count

    n_epochs = sum(e["n_epochs"] for e in per_subject.values())
    n_kept = sum(e["n_kept"] for e in per_subject.values())
    manifest = {
        "config_hash": digest,
        "config": cfg.to_dict(),
        "n_subjects": len(per_subject),
        "n_epochs": n_epochs,
        "n_kept": n_kept,
        "kept_fraction": n_kept / n_epochs if n_epochs else 0.0,
        "class_counts": totals,
        "subjects": per_subject,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nsubjects  : {manifest['n_subjects']}")
    print(f"epochs    : {n_kept} kept of {n_epochs} ({manifest['kept_fraction']:.1%})")
    print(f"classes   : {totals}")

    # Sanity gates. Both conditions below mean the build produced something
    # plausible-looking but wrong, which no later stage would flag.
    usable = {k: v for k, v in totals.items() if k != "Invalid"}
    if usable and max(usable.values()) < 0.3 * sum(usable.values()):
        print(
            "\nWARNING: no class exceeds 30% of the total. Real nights are "
            "dominated by Light sleep; a flat histogram usually means the "
            "labels are misaligned against the signal.",
            file=sys.stderr,
        )
    if manifest["kept_fraction"] < 0.7:
        print(
            f"\nWARNING: only {manifest['kept_fraction']:.1%} of epochs survived "
            "the quality gate. Check min_epoch_coverage and max_gap_seconds "
            "before training on this cache.",
            file=sys.stderr,
        )
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
