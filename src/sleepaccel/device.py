"""Torch device selection.

Why this exists: this project is developed on an Apple Silicon laptop (MPS)
but may later run on a CUDA box (a rented GPU, a lab machine). Every script
needs the same preference order and the same auditable record of what
device a run actually used, so device selection lives in one place instead
of being re-implemented (inconsistently) in every script.

MPS requires checking both `is_built()` (was torch compiled with MPS
support) and `is_available()` (is an MPS device actually present) --
`is_available()` alone can be True on a build that lacks working MPS
kernels for certain ops, and checking only `is_built()` would select MPS on
a machine with no GPU. Checking both is the documented-safe pattern.
"""

from __future__ import annotations

import platform
import sys

import torch


def select_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Args:
        prefer: optional explicit device type ("cuda", "mps", "cpu"). When
            given, it is honored as long as that backend is actually
            available; this lets tests and debugging force CPU regardless
            of what hardware is present.

    Preference order when `prefer` is None: cuda, then mps, then cpu.
    """
    if prefer is not None:
        if prefer == "cpu":
            return torch.device("cpu")
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        # Requested backend unavailable: fall through to auto-detection
        # rather than raising, so a config written for a CUDA box still
        # runs (on CPU) on a laptop.

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict:
    """Return a JSON-serializable description of the device and environment.

    This is written into every run's results JSON so a metric can later be
    traced back to the hardware and software stack that produced it --
    device backends (especially MPS) are not bit-reproducible, so this is
    part of the audit trail rather than a nicety.
    """
    return {
        "device_type": device.type,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
