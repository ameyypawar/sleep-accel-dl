"""Experiment configuration.

Why this exists: every experiment (input variant, windowing, CV scheme,
training hyperparameters) needs to be fully specified in one place so a
result can be reproduced from its config file alone, and so sweeps can be
driven by generating config variants rather than editing code.

The config is split conceptually into two groups, which matters for
caching (see `config_hash` below):

  1. Fields that change what the *preprocessed epoch tensors* look like
     (epoch_seconds, resample_hz, min_epoch_coverage, max_gap_seconds).
     Changing any of these invalidates the on-disk epoch cache.
  2. Everything else -- input variant selection, model architecture,
     training hyperparameters -- which only affects training and can reuse
     the same cached epochs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

_INPUT_VARIANTS = ("accel_only", "accel_hr", "hr_only")
_CONTEXT_MODELS = ("bilstm", "transformer", "none")
_IMBALANCE_STRATEGIES = ("none", "class_weights", "balanced_sampler")
_CV_SCHEMES = ("grouped_5fold", "loso")

# Fields that affect the cached epoch tensors -- and ONLY these. Used by
# config_hash() to key the on-disk cache. Deliberately excludes training
# knobs (batch_size, lr, weight_decay, max_epochs, patience, hidden_dim,
# dropout), model choice (context_model, context_len), and cv/seed/
# imbalance settings: all of those change how cached epochs are consumed
# during training, not what the cached epochs contain. Including them in
# the hash would invalidate (and force recomputation of) the entire epoch
# cache on every hyperparameter tweak, which is expensive and pointless --
# the windowed, resampled, coverage-filtered epoch tensors are identical
# regardless of what learning rate is used to train on them.
_CACHE_KEY_FIELDS = ("epoch_seconds", "resample_hz", "min_epoch_coverage", "max_gap_seconds")


@dataclass(frozen=True)
class ExperimentConfig:
    """Full specification of one experiment.

    See module docstring for why `_CACHE_KEY_FIELDS` is a strict subset of
    these fields rather than all of them.
    """

    input_variant: str = "accel_only"
    epoch_seconds: int = 30
    resample_hz: int = 30
    context_len: int = 20
    context_model: str = "bilstm"
    train_stride: int = 5
    imbalance_strategy: str = "class_weights"
    cv_scheme: str = "grouped_5fold"
    seed: int = 0
    min_epoch_coverage: float = 0.5
    max_gap_seconds: float = 5.0

    # Training knobs.
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    hidden_dim: int = 64
    dropout: float = 0.3

    def __post_init__(self) -> None:
        _validate_choice("input_variant", self.input_variant, _INPUT_VARIANTS)
        _validate_choice("context_model", self.context_model, _CONTEXT_MODELS)
        _validate_choice("imbalance_strategy", self.imbalance_strategy, _IMBALANCE_STRATEGIES)
        _validate_choice("cv_scheme", self.cv_scheme, _CV_SCHEMES)

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_choice(field_name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name}={value!r} is not valid; must be one of {allowed}")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file.

    Only keys matching declared dataclass fields are accepted -- an unknown
    key in the YAML raises a TypeError from the dataclass constructor,
    which is preferable to silently ignoring a typo'd field name.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig(**raw)


def config_hash(cfg: ExperimentConfig) -> str:
    """Short, stable hash of the cache-relevant subset of `cfg`.

    Only `_CACHE_KEY_FIELDS` are hashed (see module docstring): this hash
    is used to name the epoch cache directory, and must be identical for
    two configs that differ only in training hyperparameters.
    """
    cfg_dict = cfg.to_dict()
    cache_relevant = {k: cfg_dict[k] for k in _CACHE_KEY_FIELDS}
    payload = json.dumps(cache_relevant, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


# Sanity check that _CACHE_KEY_FIELDS only references real fields, caught
# at import time rather than at first call.
_known_fields = {f.name for f in fields(ExperimentConfig)}
assert set(_CACHE_KEY_FIELDS) <= _known_fields, "config_hash references unknown field(s)"
