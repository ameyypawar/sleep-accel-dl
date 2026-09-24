"""Subject-wise cross-validation folds.

This is the most important correctness property in the project. A night of
sleep yields roughly a thousand 30-second epochs from one person, and
consecutive epochs from the same person are enormously more similar to each
other than to any other subject's. Split epoch-wise and a model can score a
high kappa purely by recognising whose wrist it is looking at, having learned
nothing transferable about sleep. The result looks excellent and means
nothing.

So every split here is by subject, and :func:`assert_no_subject_leakage` is
called at the start of every training run rather than only in tests -- a
guard that only runs under pytest is a guard that will eventually be bypassed.

The split is three-way, not two-way. Early stopping needs a held-out signal,
and taking it from the test subjects leaks them into model selection just as
surely as training on them would. So the validation subjects are carved out of
the training portion, and the test subjects are touched exactly once.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCHEMES = ("grouped_5fold", "loso")


class SubjectLeakageError(AssertionError):
    """Raised when a subject appears on both sides of a split."""


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold, defined purely by subject membership."""

    index: int
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]

    @property
    def all_subjects(self) -> tuple[str, ...]:
        return self.train_subjects + self.val_subjects + self.test_subjects


def subject_folds(
    subject_ids: list[str],
    scheme: str = "grouped_5fold",
    seed: int = 0,
    n_folds: int = 5,
    n_val_subjects: int = 4,
) -> list[Fold]:
    """Build subject-disjoint folds.

    Args:
        subject_ids: every subject available.
        scheme: ``"grouped_5fold"`` or ``"loso"``. Leave-one-subject-out gives
            the lowest-variance estimate but costs one training run per
            subject, which multiplies by the number of ablation arms; it is
            the opt-in overnight setting, not the default.
        seed: controls the shuffle, so folds are reproducible.
        n_folds: number of folds when grouped.
        n_val_subjects: how many training subjects to reserve for early
            stopping.

    Returns:
        A list of :class:`Fold`, already checked for leakage.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")

    subjects = sorted(set(subject_ids))
    if len(subjects) < 3:
        raise ValueError(f"need at least 3 subjects to split, got {len(subjects)}")

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(subjects))

    if scheme == "loso":
        test_groups = [[s] for s in shuffled]
    else:
        test_groups = [list(chunk) for chunk in np.array_split(shuffled, n_folds)]

    folds: list[Fold] = []
    for index, test_subjects in enumerate(test_groups):
        remaining = [s for s in shuffled if s not in set(test_subjects)]
        # Rotate which subjects serve as validation so a single unusual
        # sleeper cannot dominate early stopping in every fold.
        offset = (index * max(n_val_subjects, 1)) % len(remaining)
        rotated = remaining[offset:] + remaining[:offset]

        n_val = min(n_val_subjects, max(1, len(rotated) - 1))
        val_subjects = rotated[:n_val]
        train_subjects = rotated[n_val:]

        folds.append(
            Fold(
                index=index,
                train_subjects=tuple(sorted(train_subjects)),
                val_subjects=tuple(sorted(val_subjects)),
                test_subjects=tuple(sorted(test_subjects)),
            )
        )

    assert_no_subject_leakage(folds)
    return folds


def assert_no_subject_leakage(folds: list[Fold]) -> None:
    """Fail loudly if any subject sits on two sides of the same fold.

    Called from the training entry point, not just from tests. Leakage does
    not crash anything and barely changes the loss curve -- it only inflates
    the final number, which is the hardest kind of bug to notice.
    """
    for fold in folds:
        train = set(fold.train_subjects)
        val = set(fold.val_subjects)
        test = set(fold.test_subjects)

        for name_a, a, name_b, b in (
            ("train", train, "val", val),
            ("train", train, "test", test),
            ("val", val, "test", test),
        ):
            overlap = a & b
            if overlap:
                raise SubjectLeakageError(
                    f"fold {fold.index}: subjects {sorted(overlap)} appear in "
                    f"both {name_a} and {name_b}"
                )

        if not train or not test:
            raise SubjectLeakageError(
                f"fold {fold.index}: empty train or test split "
                f"({len(train)} train, {len(test)} test)"
            )


def assert_test_coverage(folds: list[Fold], subject_ids: list[str]) -> None:
    """Check every subject is tested exactly once across the folds.

    A subject silently missing from all test sets would be trained on and
    never evaluated, quietly shrinking the evaluation set.
    """
    tested: list[str] = []
    for fold in folds:
        tested.extend(fold.test_subjects)

    duplicates = {s for s in tested if tested.count(s) > 1}
    if duplicates:
        raise SubjectLeakageError(f"subjects tested more than once: {sorted(duplicates)}")

    missing = set(subject_ids) - set(tested)
    if missing:
        raise SubjectLeakageError(f"subjects never tested: {sorted(missing)}")


def save_folds(folds: list[Fold], path: str | Path) -> Path:
    """Serialise folds so every experiment provably uses identical splits.

    RQ1, the RQ2 ablation arms and the RQ3 baselines must all be scored on the
    same partition, or the comparisons between them are meaningless. Writing
    the folds once and reading them back is what makes that checkable rather
    than assumed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(fold) for fold in folds], indent=2), encoding="utf-8"
    )
    return path


def load_folds(path: str | Path) -> list[Fold]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = [
        Fold(
            index=entry["index"],
            train_subjects=tuple(entry["train_subjects"]),
            val_subjects=tuple(entry["val_subjects"]),
            test_subjects=tuple(entry["test_subjects"]),
        )
        for entry in raw
    ]
    assert_no_subject_leakage(folds)
    return folds
