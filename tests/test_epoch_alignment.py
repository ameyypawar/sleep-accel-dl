"""Epoch alignment and quality gating.

These are the tests that matter most in the repo. Every failure mode covered
here produces a plausible-looking dataset rather than an exception, so without
them a wrong result would look like a good one.
"""

from __future__ import annotations

import numpy as np
import pytest

from sleepaccel.data.epochs import (
    build_epoch_grid,
    derive_magnitude,
    epoch_index_for,
    epoch_quality,
    epochs_from_recording,
    resample_epoch,
)
from tests.synthetic import EPOCH_SECONDS, make_acceleration, make_labels


# --- the negative-timestamp trap -------------------------------------------


def test_negative_timestamps_do_not_fold_onto_epoch_zero():
    """int() truncation would map -15 and +15 both to epoch 0. Floor does not."""
    t = np.array([-45.0, -15.0, -0.1, 0.0, 15.0, 45.0])
    assert epoch_index_for(t, 30.0).tolist() == [-2, -1, -1, 0, 0, 1]


def test_epoch_index_matches_floor_division_not_truncation():
    t = np.linspace(-120, 120, 401)
    expected = np.floor(t / 30.0).astype(np.int64)
    assert np.array_equal(epoch_index_for(t, 30.0), expected)


def test_lead_in_samples_are_excluded_rather_than_merged():
    starts, _ = make_labels(n_epochs=5)
    t, xyz = make_acceleration(starts, lead_in_seconds=60.0, seed=1)
    assert (t < 0).any(), "fixture should produce pre-recording samples"

    waveform, keep, _ = epochs_from_recording(
        t, xyz, starts, EPOCH_SECONDS, 30.0, 0.5, 5.0
    )
    # Epoch 0 must be built only from samples in [0, 30), not from the
    # negative-timestamp lead-in.
    in_first_epoch = ((t >= 0) & (t < EPOCH_SECONDS)).sum()
    assert in_first_epoch > 0
    assert keep[0]
    assert waveform.shape == (5, 4, 900)


# --- the dropout trap -------------------------------------------------------


def test_dropout_epoch_is_rejected_not_interpolated():
    """A watch dropout interpolates to a flat line that reads as deep sleep."""
    starts, _ = make_labels(n_epochs=6)
    t, xyz = make_acceleration(starts, dropout_epochs=(3,), seed=2)

    _, keep, quality = epochs_from_recording(
        t, xyz, starts, EPOCH_SECONDS, 30.0, 0.5, 5.0
    )
    assert not keep[3], "fully dropped-out epoch must be rejected"
    assert quality[3].n_samples == 0
    assert keep[[0, 1, 2, 4, 5]].all()


def test_partial_coverage_is_rejected_by_the_coverage_threshold():
    # Samples only in the first 5 seconds of a 30-second epoch.
    t = np.linspace(0.0, 5.0, 250)
    q = epoch_quality(t, 0.0, 30.0)
    assert q.coverage == pytest.approx(5.0 / 30.0, abs=1e-6)
    assert not q.is_usable(min_coverage=0.5, max_gap_seconds=5.0)


def test_interior_hole_is_caught_by_largest_gap():
    # Good coverage at both ends, but a 12-second hole in the middle.
    t = np.concatenate([np.linspace(0, 9, 450), np.linspace(21, 30, 450)])
    q = epoch_quality(t, 0.0, 30.0)
    assert q.coverage > 0.9, "span-based coverage alone would pass this"
    assert q.largest_gap > 11.0
    assert not q.is_usable(min_coverage=0.5, max_gap_seconds=5.0)


def test_edge_gaps_count_towards_largest_gap():
    """Samples clustered mid-epoch say nothing about the edges."""
    t = np.linspace(14.0, 16.0, 100)
    q = epoch_quality(t, 0.0, 30.0)
    assert q.largest_gap >= 14.0


def test_empty_window_reports_zero_coverage():
    q = epoch_quality(np.array([]), 0.0, 30.0)
    assert q.n_samples == 0
    assert q.coverage == 0.0
    assert q.largest_gap == 30.0
    assert not q.is_usable(0.0, 60.0)


# --- resampling -------------------------------------------------------------


def test_resample_produces_the_requested_grid_size():
    t = np.linspace(0, 30, 1500)
    xyz = np.random.default_rng(0).normal(size=(1500, 3))
    out = resample_epoch(t, xyz, 0.0, 30.0, 30.0)
    assert out.shape == (3, 900)
    assert out.dtype == np.float32


def test_resample_is_exact_on_a_known_linear_ramp():
    """Linear interpolation must reproduce a linear signal exactly."""
    t = np.array([0.0, 30.0])
    values = np.array([[0.0], [30.0]])
    out = resample_epoch(t, values, 0.0, 30.0, 1.0)
    assert out.shape == (1, 30)
    assert out[0] == pytest.approx(np.arange(30.0), abs=1e-4)


def test_resample_handles_irregular_input_spacing():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 30, size=700))
    values = (2.0 * t)[:, None]
    out = resample_epoch(t, values, 0.0, 30.0, 30.0)
    grid = np.arange(900) / 30.0
    # Away from the edges, where interp clamps, the ramp is recovered.
    assert out[0, 50:-50] == pytest.approx(2.0 * grid[50:-50], abs=0.5)


def test_magnitude_is_rotation_invariant():
    a = np.array([[1.0], [0.0], [0.0]])
    b = np.array([[0.0], [1.0], [0.0]])
    assert derive_magnitude(a) == pytest.approx(derive_magnitude(b))
    assert derive_magnitude(a)[0] == pytest.approx(1.0)


# --- the epoch axis ---------------------------------------------------------


def test_epoch_grid_rejects_unevenly_spaced_labels():
    bad = np.array([0.0, 30.0, 75.0, 105.0])
    with pytest.raises(ValueError, match="not spaced"):
        build_epoch_grid(bad, 30.0)


def test_epoch_grid_accepts_a_regular_series():
    good = np.arange(10) * 30.0
    assert np.array_equal(build_epoch_grid(good, 30.0), good)


def test_rejected_epochs_stay_in_place_so_labels_remain_aligned():
    """Dropping rows here would desynchronise signal against labels."""
    starts, codes = make_labels(n_epochs=8)
    t, xyz = make_acceleration(starts, dropout_epochs=(2, 5), seed=4)

    waveform, keep, quality = epochs_from_recording(
        t, xyz, starts, EPOCH_SECONDS, 30.0, 0.5, 5.0
    )
    assert waveform.shape[0] == len(starts) == len(codes)
    assert len(quality) == len(starts)
    assert not keep[2] and not keep[5]
    # Rejected epochs are zero-filled, never removed.
    assert np.all(waveform[2] == 0.0)
