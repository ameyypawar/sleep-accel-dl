"""Tests for sleepaccel.paths.find_dataset_root and DatasetNotFoundError.

Uses tmp_path to build fake extracted-archive trees, since the real
dataset must not be required for tests to pass.
"""

from __future__ import annotations

import pytest

from sleepaccel.paths import DatasetNotFoundError, find_dataset_root


def test_find_dataset_root_locates_nested_long_named_dir(tmp_path) -> None:
    # Mimic the real PhysioNet layout: data/raw/<long versioned name>/{motion,labels}/
    long_named_dir = (
        tmp_path
        / "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
    )
    for marker in ("motion", "labels", "heart_rate", "steps"):
        (long_named_dir / marker).mkdir(parents=True, exist_ok=True)

    found = find_dataset_root(tmp_path)

    assert found == long_named_dir


def test_partial_layout_is_not_mistaken_for_the_dataset_root(tmp_path) -> None:
    """Both markers are required: `labels/` alone is too common a name."""
    (tmp_path / "some-unrelated-project" / "labels").mkdir(parents=True)

    with pytest.raises(DatasetNotFoundError):
        find_dataset_root(tmp_path)


def test_find_dataset_root_raises_on_empty_dir(tmp_path) -> None:
    with pytest.raises(DatasetNotFoundError) as excinfo:
        find_dataset_root(tmp_path)

    assert "unzip" in str(excinfo.value)


def test_find_dataset_root_raises_when_no_marker_present(tmp_path) -> None:
    (tmp_path / "some_other_dir" / "not_the_marker").mkdir(parents=True)

    with pytest.raises(DatasetNotFoundError) as excinfo:
        find_dataset_root(tmp_path)

    assert "unzip" in str(excinfo.value)
