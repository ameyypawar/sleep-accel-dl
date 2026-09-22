"""sleepaccel: accelerometer-only sleep staging.

Package layout:
    device.py   -- torch device selection and reporting (cuda/mps/cpu).
    seed.py     -- reproducibility helpers (random/numpy/torch seeding).
    paths.py    -- dataset discovery and repo-relative path helpers.
    config.py   -- experiment configuration (dataclass, YAML load, cache hash).

Modules under this package must not assume the raw dataset is present on
disk; dataset-dependent code paths fail loudly via paths.DatasetNotFoundError
rather than silently degrading, since a missing/partial download is the
normal state during early development.
"""

from __future__ import annotations
