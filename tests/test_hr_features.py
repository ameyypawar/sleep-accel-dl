"""Per-epoch heart-rate aggregation, with emphasis on missing data."""

from __future__ import annotations

import numpy as np
import pytest

from sleepaccel.data.hr_features import (
    HR_FEATURE_NAMES,
    N_HR_FEATURES,
    epoch_hr_features,
    normalize_hr_features,
    subject_hr_baseline,
)
from tests.synthetic import EPOCH_SECONDS, make_heart_rate, make_labels


def test_feature_shape_and_names_agree():
    starts, _ = make_labels(n_epochs=10)
    hr_t, hr_bpm = make_heart_rate(starts)
    features, valid = epoch_hr_features(hr_t, hr_bpm, starts, EPOCH_SECONDS)
    assert features.shape == (10, N_HR_FEATURES)
    assert len(HR_FEATURE_NAMES) == N_HR_FEATURES
    assert valid.shape == (10,)
    assert features.dtype == np.float32


def test_epoch_without_readings_is_flagged_not_zero_bpm():
    """0 bpm is physiologically extreme, not a neutral 'missing' value."""
    starts = np.array([0.0, 30.0, 60.0])
    # Readings only in the first and last epoch.
    hr_t = np.array([1.0, 2.0, 61.0, 62.0])
    hr_bpm = np.array([60.0, 62.0, 58.0, 59.0])

    features, valid = epoch_hr_features(hr_t, hr_bpm, starts, EPOCH_SECONDS)
    assert valid.tolist() == [True, False, True]
    assert np.all(features[1] == 0.0)


def test_completely_absent_heart_rate_yields_all_invalid():
    starts, _ = make_labels(n_epochs=5)
    features, valid = epoch_hr_features(
        np.array([]), np.array([]), starts, EPOCH_SECONDS
    )
    assert not valid.any()
    assert np.all(features == 0.0)


def test_aggregates_match_a_hand_computed_window():
    starts = np.array([0.0])
    hr_t = np.array([1.0, 2.0, 3.0, 4.0])
    hr_bpm = np.array([50.0, 60.0, 70.0, 80.0])

    features, valid = epoch_hr_features(
        hr_t, hr_bpm, starts, EPOCH_SECONDS, baseline=60.0
    )
    assert valid[0]
    mean, std, lo, hi, delta = features[0]
    assert mean == pytest.approx(65.0)
    assert std == pytest.approx(np.std(hr_bpm), rel=1e-5)
    assert lo == pytest.approx(50.0)
    assert hi == pytest.approx(80.0)
    assert delta == pytest.approx(5.0)


def test_single_reading_gives_zero_std_not_nan():
    features, valid = epoch_hr_features(
        np.array([1.0]), np.array([65.0]), np.array([0.0]), EPOCH_SECONDS
    )
    assert valid[0]
    assert np.isfinite(features).all()
    assert features[0, 1] == 0.0


def test_non_finite_readings_are_ignored():
    hr_t = np.array([1.0, 2.0, 3.0])
    hr_bpm = np.array([60.0, np.nan, 70.0])
    features, valid = epoch_hr_features(
        hr_t, hr_bpm, np.array([0.0]), EPOCH_SECONDS, baseline=0.0
    )
    assert valid[0]
    assert features[0, 0] == pytest.approx(65.0)
    assert np.isfinite(features).all()


def test_baseline_is_the_median_and_resists_spikes():
    bpm = np.array([58.0, 60.0, 62.0, 61.0, 190.0])  # one awakening spike
    assert subject_hr_baseline(bpm) == pytest.approx(61.0)


def test_baseline_of_empty_recording_is_zero_not_nan():
    assert subject_hr_baseline(np.array([])) == 0.0


def test_samples_outside_the_window_are_excluded():
    starts = np.array([30.0])
    hr_t = np.array([29.9, 30.1, 59.9, 60.1])
    hr_bpm = np.array([1.0, 100.0, 200.0, 999.0])
    features, _ = epoch_hr_features(hr_t, hr_bpm, starts, EPOCH_SECONDS, baseline=0.0)
    # Only the two readings inside [30, 60) count.
    assert features[0, 2] == pytest.approx(100.0)
    assert features[0, 3] == pytest.approx(200.0)


def test_normalization_is_safe_against_zero_variance_columns():
    features = np.ones((4, N_HR_FEATURES), dtype=np.float32)
    out = normalize_hr_features(features, mean=np.ones(N_HR_FEATURES), std=np.zeros(N_HR_FEATURES))
    assert np.isfinite(out).all()
    assert np.all(out == 0.0)
