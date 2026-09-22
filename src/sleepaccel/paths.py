"""Dataset discovery and repo-relative path helpers.

Why this exists: the PhysioNet archive for this dataset (Walch et al. 2019,
"sleep-accel") does NOT extract to a directory named `sleep-accel` -- it
extracts to a long, versioned directory name
(`motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-
polysomnography-1.0.0`), and that name could change across dataset
versions. Hardcoding it would silently break the moment PhysioNet bumps a
version number. Instead we discover the real dataset root by structure --
it is, by construction, the directory that has a `labeled_sleep` child --
which is stable regardless of the outer directory's name.

This module must import cleanly and be fully testable with no dataset
present: every dataset-dependent function fails with `DatasetNotFoundError`
rather than returning something misleading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"

# Child directory name that marks a directory as the dataset root.
_MARKER_DIRNAME = "labeled_sleep"


class DatasetNotFoundError(Exception):
    """Raised when the raw dataset cannot be located under `data/raw`.

    The message always includes the exact extraction command so a fresh
    contributor (or Phase 0 script run before the download finishes) gets
    an actionable fix rather than a bare "not found".
    """


def _not_found_message(raw_dir: Path) -> str:
    return (
        f"Dataset not found under {raw_dir}. Expected a directory (at "
        f"any depth up to {3} levels) containing a '{_MARKER_DIRNAME}' "
        "subdirectory -- this is how the PhysioNet archive is identified, "
        "since its extracted directory name is long and version-dependent.\n"
        "If the archive has been downloaded but not extracted, run:\n"
        f"  unzip {raw_dir}/sleep-accel.zip -d {raw_dir}\n"
        "then re-run this script."
    )


def find_dataset_root(raw_dir: Path, max_depth: int = 3) -> Path:
    """Locate the extracted dataset root under `raw_dir`.

    Searches recursively, up to `max_depth` levels below `raw_dir`, for a
    directory containing a `labeled_sleep` child -- that structural marker
    is what identifies the true dataset root, independent of the outer
    directory's (long, version-dependent) name.

    Raises:
        DatasetNotFoundError: if `raw_dir` does not exist, is empty, or no
            matching directory is found within `max_depth` levels.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir() or not any(raw_dir.iterdir()):
        raise DatasetNotFoundError(_not_found_message(raw_dir))

    if (raw_dir / _MARKER_DIRNAME).is_dir():
        return raw_dir

    candidates: list[Path] = []

    def _walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            subdirs = sorted(p for p in directory.iterdir() if p.is_dir())
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return
        for sub in subdirs:
            if (sub / _MARKER_DIRNAME).is_dir():
                candidates.append(sub)
            else:
                _walk(sub, depth + 1)

    _walk(raw_dir, 1)

    if not candidates:
        raise DatasetNotFoundError(_not_found_message(raw_dir))

    # Prefer the shallowest match; break ties alphabetically for determinism.
    candidates.sort(key=lambda p: (len(p.relative_to(raw_dir).parts), str(p)))
    return candidates[0]


@dataclass(frozen=True)
class SubjectFiles:
    """Paths to the four raw files for one subject."""

    subject_id: str
    acceleration: Path
    heart_rate: Path
    steps: Path
    labels: Path


def cache_dir(config_hash: str) -> Path:
    """Directory for cached, preprocessed epoch tensors for a given config hash.

    Keyed by `config_hash` (see config.config_hash) rather than the full
    config so unrelated training-only changes do not invalidate the cache.
    """
    d = REPO_ROOT / "data" / "cache" / config_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(run_id: str) -> Path:
    """Directory for a single run's outputs (metrics JSON, checkpoints, plots)."""
    d = REPO_ROOT / "results" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
