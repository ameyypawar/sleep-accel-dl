"""Reproducibility helpers.

Why this exists: five-fold (or LOSO) cross-validation results are only
comparable across runs if every fold starts from the same RNG state. This
module centralizes seeding so training scripts, notebooks, and tests all
seed the same set of generators the same way.

Important caveat, documented here rather than left to be rediscovered by
whoever debugs a flaky metric: MPS training is NOT bit-reproducible even
with every generator seeded, because several MPS kernels (notably some
reduction and interpolation ops) use nondeterministic accumulation order
on Apple's GPU. Expect fold-level metrics to vary in the third decimal
place between runs on MPS with an identical seed. This is a hardware/driver
property, not a bug in this codebase. CUDA and CPU runs are much closer to
deterministic, and are bit-reproducible when `strict=True` is used (at the
cost of raising on any op without a deterministic implementation).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, strict: bool = False) -> None:
    """Seed all RNGs used by this project.

    Args:
        seed: the seed value, applied to `random`, `numpy.random`, and
            torch's CPU/CUDA/MPS generators.
        strict: when True, also call `torch.use_deterministic_algorithms(True)`.
            This is opt-in, not the default, because several ops (e.g. some
            forms of interpolation and index_put with accumulation) raise
            `RuntimeError` on the MPS backend when deterministic algorithms
            are forced, and would otherwise crash MPS-based development
            runs outright. Enable it for CPU/CUDA runs where reproducibility
            matters more than op coverage.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if strict:
        torch.use_deterministic_algorithms(True)


def dataloader_worker_init(worker_id: int) -> None:
    """Seed a DataLoader worker process so multi-worker loading is reproducible.

    Each worker inherits the parent process's RNG state via fork, which
    without this would make all workers draw identical "random" augmentation
    sequences. Derive a distinct seed per worker from the worker id and the
    current base seed.
    """
    base_seed = torch.initial_seed() % (2**32)
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
