"""PSG sleep-stage labels and their mapping to the 4-class task.

The label files score each 30-second epoch with a stage code. The dataset
documentation lists only::

    0 = Wake, 1 = N1, 2 = N2, 3 = N3, 5 = REM

The files themselves disagree. Counting every label across all 31 subjects
gives ``{-1: 438, 0: 2429, 1: 1821, 2: 12954, 3: 3329, 4: 356, 5: 5884}``.
Stage 4 is present, in 356 epochs, and ``-1`` marks unscored epochs. Neither
appears in the published description.

Stage 4 is R&K's deepest slow-wave sleep, which AASM later merged into N3, so
it maps to Deep exactly as 3 does. Trusting the documentation and treating 4
as unknown would silently discard those 356 epochs -- around 10% of the Deep
class, which is already the scarcest of the four and the hardest to learn.

The absence of a contiguous range is why this mapping is a dict rather than a
list indexed by raw code. Positional indexing would put REM (5) out of bounds,
or worse, onto whatever happened to occupy index 5.

Values outside the known set, ``-1`` included, map to :data:`INVALID` and are
flagged in the returned mask rather than guessed at. Those epochs are excluded
from both training loss and evaluation: scoring a model against a label nobody
assigned would be meaningless.
"""

from __future__ import annotations

import numpy as np

#: Sentinel for an epoch with no usable label. Chosen as -1 so it can be
#: passed straight to ``torch.nn.functional.cross_entropy(ignore_index=-1)``.
INVALID = -1

#: Raw stage code -> 4-class index. A dict, never a list: the codes are not a
#: contiguous range and positional indexing would misplace REM.
RAW_TO_CLASS4: dict[int, int] = {
    0: 0,  # Wake
    1: 1,  # N1          -> Light
    2: 1,  # N2          -> Light
    3: 2,  # N3          -> Deep
    4: 2,  # R&K stage 4 -> Deep (merged into N3 by AASM; 356 epochs present)
    5: 3,  # REM
}

CLASS_NAMES: tuple[str, ...] = ("Wake", "Light", "Deep", "REM")
N_CLASSES = len(CLASS_NAMES)

#: Binary collapse used to compare against Cole-Kripke and Sadeh, which were
#: only ever defined for sleep/wake.
BINARY_NAMES: tuple[str, ...] = ("Wake", "Sleep")


def map_labels(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map raw AASM stage codes to 4-class indices.

    Args:
        raw: integer stage codes, shape ``[N]``.

    Returns:
        ``(class4, valid)`` where ``class4`` holds indices into
        :data:`CLASS_NAMES` with :data:`INVALID` for unrecognised codes, and
        ``valid`` is a boolean mask of epochs that carry a usable label.
    """
    raw = np.asarray(raw)
    class4 = np.full(raw.shape, INVALID, dtype=np.int64)
    for raw_code, class_index in RAW_TO_CLASS4.items():
        class4[raw == raw_code] = class_index
    return class4, class4 != INVALID


def collapse_to_binary(class4: np.ndarray) -> np.ndarray:
    """Collapse 4-class labels to sleep (1) versus wake (0).

    Invalid epochs stay :data:`INVALID` so they remain excludable. This exists
    for the classical-baseline comparison: Cole-Kripke and Sadeh only produce
    sleep/wake, so comparing them against the 4-class model requires putting
    both on the binary axis.
    """
    class4 = np.asarray(class4)
    binary = np.full(class4.shape, INVALID, dtype=np.int64)
    binary[class4 == 0] = 0
    binary[np.isin(class4, (1, 2, 3))] = 1
    return binary


def class_histogram(class4: np.ndarray) -> dict[str, int]:
    """Count epochs per class, for the cache manifest and dashboard.

    A roughly uniform histogram is a red flag, not a success: real nights are
    dominated by Light sleep, with Deep and REM as clear minorities. A flat
    distribution almost always means the labels have been misaligned against
    the signal.
    """
    class4 = np.asarray(class4)
    counts = {name: int((class4 == i).sum()) for i, name in enumerate(CLASS_NAMES)}
    counts["Invalid"] = int((class4 == INVALID).sum())
    return counts
