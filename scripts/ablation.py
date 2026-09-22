"""RQ2: quantify what heart rate adds, with a paired bootstrap.

    python scripts/ablation.py

Compares the trained variants on the intersection of epochs they all
predicted, and reports the kappa difference with a 95% interval. The bootstrap
resamples subjects rather than epochs, since epochs within one night are not
independent, and applies the same resample to both arms so the comparison is
paired.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))

import argparse
import json

import numpy as np

from sleepaccel.metrics import bootstrap_kappa_delta
from sleepaccel.paths import REPO_ROOT

PAIRS = [
    # What heart rate adds to motion. The headline ablation.
    ("accel_only", "accel_hr"),
    # Heart rate alone against motion alone.
    ("accel_only", "hr_only"),
    # The crux: does the accelerometer contribute anything ON TOP of heart
    # rate? If this interval contains zero, motion is redundant once HR is
    # available, which is the strongest statement this dataset can make about
    # the original hypothesis.
    ("hr_only", "accel_hr"),
    # Does the sequence model earn its place?
    ("accel_only", "accel_only_nocontext"),
]


def load(run_id: str):
    path = REPO_ROOT / "results" / run_id / "predictions.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as handle:
        return {k: handle[k] for k in handle.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out: dict[str, dict] = {}
    for base, other in PAIRS:
        a, b = load(base), load(other)
        if a is None or b is None:
            print(f"skipping {base} vs {other}: missing predictions")
            continue

        # The arms must be compared on identical epochs. They share folds, so
        # the ordering matches, but assert rather than assume.
        if a["y_true"].shape != b["y_true"].shape or not np.array_equal(
            a["y_true"], b["y_true"]
        ):
            print(f"skipping {base} vs {other}: ground truth differs between arms")
            continue

        result = bootstrap_kappa_delta(
            a["y_true"], a["y_pred"], b["y_pred"], a["subjects"],
            n_boot=args.n_boot, seed=args.seed,
        )
        name = f"{other} - {base}"
        out[name] = result
        sig = "excludes zero" if result["excludes_zero"] else "includes zero"
        print(
            f"{name:>40}: delta kappa = {result['delta_mean']:+.4f}  "
            f"95% CI [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  ({sig})"
        )

    path = REPO_ROOT / "results" / "ablation.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
