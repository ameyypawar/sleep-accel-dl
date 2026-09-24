"""Where the context windows fall over one subject's epochs. No torch.

The torch dataset needs this arithmetic to build windows, and the dashboard
needs it to work out which epoch each saved prediction belongs to -- without
importing torch just to find out. Keeping a single copy here means the two
cannot quietly disagree about where windows start.
"""

from __future__ import annotations

import numpy as np


def window_index(n_epochs: int, context_len: int, stride: int) -> list[int]:
    """Start indices for windows over a single subject.

    The final window is pulled back to end exactly at ``n_epochs`` rather than
    being dropped, so the tail of every recording is still seen. That has a
    cost: unless ``n_epochs`` is a multiple of the stride, the last window
    re-covers epochs that the window before it already did. At evaluation,
    where the stride equals the window length, up to ``context_len - 1`` epochs
    at the end of each recording are therefore predicted twice. Measured on
    this dataset that is 284 of 25,569 evaluated epochs (1.1%), moving pooled
    kappa by at most 0.001. :func:`evaluation_epoch_order` exposes the repeats
    so they can be removed.
    """
    if n_epochs < context_len:
        return [0] if n_epochs > 0 else []

    starts = list(range(0, n_epochs - context_len + 1, stride))
    last = n_epochs - context_len
    if starts and starts[-1] != last:
        starts.append(last)
    return starts


def evaluation_epoch_order(keep: np.ndarray, context_len: int) -> np.ndarray:
    """The epoch index of each saved prediction, in the order it was emitted.

    Evaluation writes one subject's predictions as a flat sequence: window by
    window, with epochs that failed the quality gate left out, and with no
    epoch index stored alongside. This rebuilds that index. It is what lets a
    prediction be placed on the real time axis of the night, and any index
    that appears twice in the result is one of the repeated tail epochs
    described in :func:`window_index`.
    """
    keep = np.asarray(keep, dtype=bool)
    n_epochs = int(keep.size)
    order = [
        epoch
        for start in window_index(n_epochs, context_len, context_len)
        for epoch in range(start, min(start + context_len, n_epochs))
        if keep[epoch]
    ]
    return np.asarray(order, dtype=np.int64)
