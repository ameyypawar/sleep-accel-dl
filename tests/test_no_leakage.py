"""Subject-wise split integrity.

If one test in this repo should never be deleted, it is this file. Subject
leakage does not raise, does not distort the loss curve, and does not look
wrong in any plot -- it only inflates the final metric.
"""

from __future__ import annotations

import pytest

from sleepaccel.data.splits import (
    Fold,
    SubjectLeakageError,
    assert_no_subject_leakage,
    assert_test_coverage,
    load_folds,
    save_folds,
    subject_folds,
)

SUBJECTS = [f"s{i:02d}" for i in range(31)]


# --- the property that matters ---------------------------------------------


@pytest.mark.parametrize("scheme", ["grouped_5fold", "loso"])
def test_no_subject_appears_on_two_sides_of_a_fold(scheme):
    for fold in subject_folds(SUBJECTS, scheme=scheme):
        train, val, test = (
            set(fold.train_subjects),
            set(fold.val_subjects),
            set(fold.test_subjects),
        )
        assert not train & val
        assert not train & test
        assert not val & test


@pytest.mark.parametrize("scheme", ["grouped_5fold", "loso"])
def test_every_subject_is_tested_exactly_once(scheme):
    folds = subject_folds(SUBJECTS, scheme=scheme)
    assert_test_coverage(folds, SUBJECTS)


def test_validation_subjects_come_out_of_training_not_test():
    """Early stopping on test subjects leaks them into model selection."""
    for fold in subject_folds(SUBJECTS):
        assert fold.val_subjects
        assert not set(fold.val_subjects) & set(fold.test_subjects)


def test_loso_produces_one_fold_per_subject():
    folds = subject_folds(SUBJECTS, scheme="loso")
    assert len(folds) == len(SUBJECTS)
    assert all(len(fold.test_subjects) == 1 for fold in folds)


def test_grouped_scheme_produces_the_requested_fold_count():
    folds = subject_folds(SUBJECTS, scheme="grouped_5fold", n_folds=5)
    assert len(folds) == 5


def test_folds_are_reproducible_for_a_fixed_seed():
    a = subject_folds(SUBJECTS, seed=7)
    b = subject_folds(SUBJECTS, seed=7)
    assert [f.test_subjects for f in a] == [f.test_subjects for f in b]


def test_different_seeds_give_different_partitions():
    a = subject_folds(SUBJECTS, seed=0)
    b = subject_folds(SUBJECTS, seed=1)
    assert [f.test_subjects for f in a] != [f.test_subjects for f in b]


# --- the guard itself must actually fire -----------------------------------


def test_leakage_guard_catches_a_deliberately_corrupted_fold():
    bad = [
        Fold(
            index=0,
            train_subjects=("s01", "s02", "s03"),
            val_subjects=("s04",),
            test_subjects=("s02",),  # also in train
        )
    ]
    with pytest.raises(SubjectLeakageError, match="s02"):
        assert_no_subject_leakage(bad)


def test_leakage_guard_catches_val_test_overlap():
    bad = [
        Fold(
            index=3,
            train_subjects=("s01",),
            val_subjects=("s09",),
            test_subjects=("s09",),
        )
    ]
    with pytest.raises(SubjectLeakageError, match="val and test"):
        assert_no_subject_leakage(bad)


def test_empty_test_split_is_rejected():
    bad = [Fold(index=0, train_subjects=("s01",), val_subjects=(), test_subjects=())]
    with pytest.raises(SubjectLeakageError, match="empty"):
        assert_no_subject_leakage(bad)


def test_duplicate_test_membership_is_rejected():
    folds = [
        Fold(0, ("s03",), ("s04",), ("s01",)),
        Fold(1, ("s03",), ("s04",), ("s01",)),
    ]
    with pytest.raises(SubjectLeakageError, match="more than once"):
        assert_test_coverage(folds, ["s01", "s03", "s04"])


def test_untested_subject_is_rejected():
    folds = [Fold(0, ("s02",), ("s03",), ("s01",))]
    with pytest.raises(SubjectLeakageError, match="never tested"):
        assert_test_coverage(folds, ["s01", "s02", "s03", "s99"])


# --- persistence ------------------------------------------------------------


def test_folds_round_trip_through_disk(tmp_path):
    """Every experiment must provably use the same partition."""
    folds = subject_folds(SUBJECTS, seed=3)
    path = save_folds(folds, tmp_path / "folds.json")
    restored = load_folds(path)
    assert restored == folds


def test_too_few_subjects_is_rejected():
    with pytest.raises(ValueError, match="at least 3"):
        subject_folds(["s01", "s02"])


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="scheme must be"):
        subject_folds(SUBJECTS, scheme="random_epochs")
