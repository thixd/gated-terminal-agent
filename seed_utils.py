#!/usr/bin/env python3
"""
Utilities for reproducible experiment seeding.
"""

from __future__ import annotations

import os
import random

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional during early setup
    np = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional during early setup
    torch = None


def set_global_seed(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs and enable deterministic settings.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
