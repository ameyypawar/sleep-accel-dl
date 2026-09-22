"""Tests for sleepaccel.device, sleepaccel.seed, and the config/cache-hash split.

These must all pass with no dataset present -- they exercise device
selection, RNG seeding, and config validation only.
"""

from __future__ import annotations

import torch

from sleepaccel.config import ExperimentConfig, config_hash
from sleepaccel.device import describe_device, select_device
from sleepaccel.seed import seed_everything


def test_select_device_returns_torch_device() -> None:
    device = select_device()
    assert isinstance(device, torch.device)


def test_select_device_honours_prefer_cpu() -> None:
    device = select_device(prefer="cpu")
    assert device.type == "cpu"


def test_describe_device_has_expected_keys() -> None:
    device = select_device(prefer="cpu")
    info = describe_device(device)
    assert set(info.keys()) == {"device_type", "torch_version", "python_version", "platform"}
    assert info["device_type"] == "cpu"


def test_seed_everything_makes_torch_randn_reproducible() -> None:
    seed_everything(1234)
    a1 = torch.randn(4)
    a2 = torch.randn(4)

    seed_everything(1234)
    b1 = torch.randn(4)
    b2 = torch.randn(4)

    assert torch.equal(a1, b1)
    assert torch.equal(a2, b2)


def test_config_rejects_bad_input_variant() -> None:
    try:
        ExperimentConfig(input_variant="bogus")
    except ValueError as e:
        assert "input_variant" in str(e)
    else:
        raise AssertionError("expected ValueError for bad input_variant")


def test_config_rejects_bad_cv_scheme() -> None:
    try:
        ExperimentConfig(cv_scheme="bogus")
    except ValueError as e:
        assert "cv_scheme" in str(e)
    else:
        raise AssertionError("expected ValueError for bad cv_scheme")


def test_config_hash_stable_across_identical_configs() -> None:
    cfg_a = ExperimentConfig()
    cfg_b = ExperimentConfig()
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_config_hash_changes_with_resample_hz() -> None:
    cfg_a = ExperimentConfig(resample_hz=30)
    cfg_b = ExperimentConfig(resample_hz=50)
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_config_hash_does_not_change_with_lr() -> None:
    # This is the point of the split: lr is a training knob, not a
    # cache-relevant field, so changing it must not invalidate the epoch
    # cache.
    cfg_a = ExperimentConfig(lr=1e-3)
    cfg_b = ExperimentConfig(lr=1e-1)
    assert config_hash(cfg_a) == config_hash(cfg_b)
