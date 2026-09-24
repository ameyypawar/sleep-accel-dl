"""Window placement and the reconstructed evaluation order.

The dashboard places saved predictions on the night's time axis using
evaluation_epoch_order, so if it disagreed with how evaluation really walked
the windows, every hypnogram would be silently shifted. These pin it down.
"""

from __future__ import annotations

import numpy as np

from sleepaccel.data.windows import evaluation_epoch_order, window_index


def test_last_window_is_pulled_back_to_the_end():
    # 45 epochs in windows of 20: 0-19, 20-39, then 25-44 rather than dropping 40-44.
    assert window_index(45, 20, 20) == [0, 20, 25]


def test_evaluation_repeats_exactly_the_overlapped_tail():
    order = evaluation_epoch_order(np.ones(45, dtype=bool), context_len=20)
    assert order.size == 60
    assert np.array_equal(np.unique(order), np.arange(45))
    repeated = np.flatnonzero(np.bincount(order) > 1)
    assert np.array_equal(repeated, np.arange(25, 40))


def test_exact_multiple_has_no_repeats():
    order = evaluation_epoch_order(np.ones(40, dtype=bool), context_len=20)
    assert np.array_equal(order, np.arange(40))


def test_rejected_epochs_are_left_out_in_place():
    keep = np.ones(40, dtype=bool)
    keep[[3, 21]] = False
    order = evaluation_epoch_order(keep, context_len=20)
    assert 3 not in order and 21 not in order
    assert order.size == 38
    assert np.all(np.diff(order) > 0), "no repeats here, so order must be strictly increasing"


def test_recording_shorter_than_one_window():
    assert window_index(10, 20, 20) == [0]
    assert np.array_equal(evaluation_epoch_order(np.ones(10, dtype=bool), 20), np.arange(10))


def test_empty_recording():
    assert window_index(0, 20, 20) == []
    assert evaluation_epoch_order(np.zeros(0, dtype=bool), 20).size == 0
