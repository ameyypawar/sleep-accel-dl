"""Label mapping, with emphasis on the non-contiguous stage vocabulary."""

from __future__ import annotations

import numpy as np
import pytest

from sleepaccel.data.labels import (
    CLASS_NAMES,
    INVALID,
    N_CLASSES,
    RAW_TO_CLASS4,
    class_histogram,
    collapse_to_binary,
    map_labels,
)


def test_all_valid_codes_map_into_range():
    raw = np.array([0, 1, 2, 3, 5])
    class4, valid = map_labels(raw)
    assert valid.all()
    assert class4.tolist() == [0, 1, 1, 2, 3]
    assert class4.max() < N_CLASSES


def test_stage_four_is_not_a_valid_code():
    """There is no stage 4 in AASM scoring; it must not silently map."""
    assert 4 not in RAW_TO_CLASS4
    class4, valid = map_labels(np.array([4]))
    assert class4.tolist() == [INVALID]
    assert not valid.any()


def test_rem_is_five_not_four():
    """The trap: a list indexed by raw code would put REM at the wrong slot."""
    class4, _ = map_labels(np.array([5]))
    assert CLASS_NAMES[int(class4[0])] == "REM"


def test_unknown_and_sentinel_codes_are_invalid():
    class4, valid = map_labels(np.array([-1, 6, 99]))
    assert (class4 == INVALID).all()
    assert not valid.any()


def test_n1_and_n2_both_collapse_to_light():
    class4, _ = map_labels(np.array([1, 2]))
    assert class4.tolist() == [1, 1]
    assert CLASS_NAMES[1] == "Light"


def test_binary_collapse_preserves_invalid():
    class4 = np.array([0, 1, 2, 3, INVALID])
    binary = collapse_to_binary(class4)
    assert binary.tolist() == [0, 1, 1, 1, INVALID]


def test_class_histogram_counts_every_class_and_invalid():
    class4 = np.array([0, 0, 1, 2, 3, INVALID])
    hist = class_histogram(class4)
    assert hist == {"Wake": 2, "Light": 1, "Deep": 1, "REM": 1, "Invalid": 1}


def test_mapping_is_pure_and_does_not_mutate_input():
    raw = np.array([0, 1, 5])
    before = raw.copy()
    map_labels(raw)
    assert np.array_equal(raw, before)


@pytest.mark.parametrize("code,expected", [(0, "Wake"), (1, "Light"), (2, "Light"), (3, "Deep"), (5, "REM")])
def test_each_code_maps_to_the_documented_name(code, expected):
    class4, _ = map_labels(np.array([code]))
    assert CLASS_NAMES[int(class4[0])] == expected
